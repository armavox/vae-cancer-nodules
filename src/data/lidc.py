import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pickle
import pylidc
from pylidc.utils import consensus
from scipy.stats import mode
from scipy.ndimage.morphology import binary_dilation
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from data.helpers import extract_cube
from data.transforms import Normalization
from utils.helpers import config_snapshot


logger = logging.getLogger("LIDCNodulesDataset")


nodule_features_list = [
    "calcification",
    "internalStructure",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
]


@dataclass
class LIDCNodule:
    pylidc_scan: pylidc.Scan
    bbox: Tuple[slice]
    mask: np.ndarray
    centroid: np.ndarray
    diameter: float
    texture: int


class LIDCNodulesDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        cube_voxelsize: int = 48,
        extract_size_mm: float = 48.0,
        nodule_diameter_interval: list = [10, 30],
        annotations_number_required: int = 3,
        masked: bool = False,
        mask_dilation_iters: int = 12,
        cut_denom: int = 3,
        cluster_list_pickle_path: str = "./src/data/aux/lidc_cluster_list.pickle",
        nodule_list_pickle_path: str = "./src/data/aux/lidc_nodule_list.pickle",
        ct_clip_range: List[int] = [-1000, 600],
        mapping_range: List[float] = [-0.25, 0.75],
    ):
        """Pytorch dataset class for exctracting LIDC-IDRI dataset nodules.

        Parameters
        ----------
        datapath : str
            Path to LIDC-IDRI dataset with folders LIDC-IDRI-**** for each DICOM research.
            Folder should have old LIDC-IDRI structure (for more info look pylidc library doc).
        cube_voxelsize : int, optional
            Shape of extracted nodule cubes, by default 48
        extract_size_mm : float, optional
            Actual size in mm of extracted cube around nodule, by default 48.0
        hist_eq_norm_json_filepath : str, optional
            Path to file to save/load normalization characteristics,
            by default "./src/data/aux/lidc_histeq_norm_stats.json"
        nodule_diameter_interval : tuple, optional
            All extracted nodules will have diameters in the provided interval,
            by default [10, 30)
        annotations_number_required : int, optional
            Number of annotators of the nodule for aquire nodule characteristics in una,biguous way,
            by default 3
        mask_dilation_iters : int, optional
            Argument for `scipy.ndimage.binary_dilation`. Defines size of dilated mask for the nodule,
            by default 12
        cut_denom : int, optional
            Denominator for diameter of sphere to cut nodule center. To return masked_nodule,
            nodule center is cutted by nodule.diameter / cut_denom.
            by default 3
        cluster_list_pickle_path : str, optional
            Auxiliary file for faster dataset loading in second and subsequent runs,
            by default "./src/data/aux/lidc_cluster_list.pickle"
        nodule_list_pickle_path : str, optional
            Auxiliary file for faster dataset loading in second and subsequent runs,
            by default "./src/data/aux/lidc_nodule_list.pickle"
        composed: bool, optional
            If Dataset used as part of composed dataset (see ctln_dataset.LIDC_LNDb_Dataset),
            by default False
        """

        self.datapath = datapath
        self.write_pylidcrc(self.datapath)
        self.cube_voxelsize = cube_voxelsize
        self.extract_size_mm = extract_size_mm
        self.diam_interval = nodule_diameter_interval
        self.annotations_number_required = annotations_number_required
        self.cluster_list_pickle_path = cluster_list_pickle_path
        self.nodule_list_pickle_path = nodule_list_pickle_path
        self.masked = masked
        self.mask_dilation_iters = mask_dilation_iters
        self.cut_denom = cut_denom

        cluster_list = self.__prepare_nodules_annotations()
        self.nodule_list = self.__prepare_nodule_list(cluster_list)

        self.clip_range = ct_clip_range
        self.norm = Normalization(
            from_min=self.clip_range[0],
            from_max=self.clip_range[1],
            to_min=mapping_range[0],
            to_max=mapping_range[1],
        )

    def __len__(self):
        return len(self.nodule_list)

    def __getitem__(self, i):
        nodule = self.nodule_list[i]
        nodule_vol, nodule_mask = self.load_nodule_vol(nodule)
        nodule_vol = self.norm(np.clip(nodule_vol, *self.clip_range))

        sample = {  # permuted to [C, D, H, W]
            "lidc_nodule": nodule,
            "nodule": torch.from_numpy(nodule_vol).type(torch.float).unsqueeze(0).permute(0, 3, 1, 2),
            "mask": torch.from_numpy(nodule_mask).type(torch.long).permute(2, 0, 1)
        }
        return sample

    def load_nodule_vol(self, nodule: LIDCNodule):
        bb = nodule.bbox
        volume = nodule.pylidc_scan.to_volume(verbose=False)

        mask_vol = np.zeros(volume.shape)
        mask_vol[bb[0].start : bb[0].stop, bb[1].start : bb[1].stop, bb[2].start : bb[2].stop][
            nodule.mask
        ] = 1

        nodule_vol = extract_cube(
            series_volume=volume,
            spacing=nodule.pylidc_scan.spacings,
            nodule_coords=nodule.centroid,
            cube_voxelsize=self.cube_voxelsize,
            extract_size_mm=self.extract_size_mm,
        )
        nodule_levelset_vol = extract_cube(
            series_volume=mask_vol,
            spacing=nodule.pylidc_scan.spacings,
            nodule_coords=nodule.centroid,
            cube_voxelsize=self.cube_voxelsize,
            extract_size_mm=self.extract_size_mm,
        )
        return nodule_vol, nodule_levelset_vol

    def __prepare_nodules_annotations(self):
        """Search through pylidc database for annotations, make clusters
        of anns corresponged to same nodules and forms list of clusters.
        """

        # Prepare or load annotations clustered for each nodule
        lidc_ann_config = {"annotations_number_required": self.annotations_number_required}
        ann_snapshot_exists = config_snapshot(
            "lidc_ann", lidc_ann_config, "./src/data/aux/.lidcann_config_snapshot.json"
        )
        ann_pickle_exists = os.path.exists(self.cluster_list_pickle_path)
        if not ann_pickle_exists or not ann_snapshot_exists:
            cluster_list = []

            for series in tqdm(pylidc.query(pylidc.Scan).all(), desc="Preparing LIDC annotations list"):
                clusters = series.cluster_annotations(verbose=False)
                # We take only nodules with >=3 annotations for robustness.
                clusters = [c for c in clusters if len(c) >= self.annotations_number_required]
                if len(clusters) > 0:
                    cluster_list.append(clusters)
            # Flatten cluster_list
            cluster_list = [c for cl in cluster_list for c in cl]
            # Dump cluster_list for future use
            logger.info("pickling LIDC annotation list for future use")
            with open(self.cluster_list_pickle_path, "wb") as f:
                pickle.dump(cluster_list, f)
        else:
            with open(self.cluster_list_pickle_path, "rb") as f:
                cluster_list = pickle.load(f)
        return cluster_list

    def __prepare_nodule_list(self, cluster_list: List[List[pylidc.Annotation]]):
        lidc_nodule_config = {
            "diam_interval": self.diam_interval,
            "extract_size_mm": self.extract_size_mm,
            "mask_dilation_iters": self.mask_dilation_iters,
        }
        nodule_pickle_exists = os.path.exists(self.nodule_list_pickle_path)
        snapshot_exists = config_snapshot(
            "lidc_nodule", lidc_nodule_config, "./src/data/aux/.lidcnod_config_snapshot.json"
        )
        if not nodule_pickle_exists or not snapshot_exists:
            nodule_list = []
            _tqdm_kwargs = {"desc": "Preparing LIDC nodule list", "total": len(cluster_list)}
            for i, cluster in tqdm(enumerate(cluster_list), **_tqdm_kwargs):
                # Check if all annotations belong to the same scan
                if len(np.unique([ann.scan.id for ann in cluster])) != 1:
                    logger.warning(f"annotations not from the same scans! skip")
                    continue

                nodule_diam = np.mean([ann.diameter for ann in cluster])
                texture_scores = [ann.texture for ann in cluster]
                # Skip nodules out of diam interval and with amiguous texture scores
                if (
                    nodule_diam < self.diam_interval[0]
                    or nodule_diam >= self.diam_interval[1]
                    or not_valid_score(texture_scores)
                ):
                    continue

                # Minimal possible bbox size (in mm).
                minsize = max([max(cl.bbox_dims(pad=None)) for cl in cluster])
                pad_mm = max(float(self.extract_size_mm), minsize)
                nodule_mask, nodule_bbox = consensus(cluster, clevel=0.8, pad=pad_mm, ret_masks=False)
                dilated_nodule_mask = binary_dilation(nodule_mask, iterations=self.mask_dilation_iters)
                nodule_coords = np.mean([ann.centroid for ann in cluster], axis=0)
                nodule_diam = np.mean([ann.diameter for ann in cluster])
                nodule_texture = mode(texture_scores).mode.item()

                nodule = LIDCNodule(
                    pylidc_scan=cluster[0].scan,
                    bbox=nodule_bbox,
                    mask=dilated_nodule_mask,
                    centroid=nodule_coords,
                    diameter=nodule_diam,
                    texture=nodule_texture,
                )
                nodule_list.append(nodule)

            logger.info("pickling LIDC nodule list for future use")
            with open(self.nodule_list_pickle_path, "wb") as f:
                pickle.dump(nodule_list, f)
        else:
            with open(self.nodule_list_pickle_path, "rb") as f:
                nodule_list = pickle.load(f)
        return nodule_list

    @staticmethod
    def write_pylidcrc(datapath, warn=True):
        """Autocreate ~/.pylidcrc config file"""
        with open(os.path.join(os.path.expanduser("~"), ".pylidcrc"), "w") as f:
            f.write(f"[dicom]\n")
            f.write(f"path = {datapath}\n")
            f.write(f"warn = {warn}")


def not_valid_score(scores: List[int]):
    """Checks if the set of estimations is ambiguous (all scores are different)."""
    return True if len(np.unique(scores)) == len(scores) else False


if __name__ == "__main__":

    config = {
        "datapath": "/data/ssd2/ctln-gan-data/LIDC-IDRI",
        "cube_voxelsize": 64,
        "extract_size_mm": 64.0,
        "nodule_diameter_interval": [8.0, 30.0],
        "masked": False,
        "ct_clip_range": (-1000, 600),
        "mapping_range": [-1.0, 1.0]
    }
    dataset = LIDCNodulesDataset(**config)

    # from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ws_path = "/home/artem.lobantsev/ssd/rls-med-ws/tensorboard_logs2"
    writer = SummaryWriter(os.path.join(ws_path, "lidc-log0"))

    norm = Normalization(from_min=-1.0, from_max=1.0, to_min=0, to_max=255)
    norm2 = Normalization(from_min=-1000, from_max=600, to_min=0, to_max=1)
    for i, sample in tqdm(enumerate(dataset)):
        patient_id = sample["lidc_nodule"].pylidc_scan.patient_id
        scan = sample["lidc_nodule"].pylidc_scan.to_volume()
        clip_scan = np.clip(scan, *config["ct_clip_range"])
        img = dataset.norm.denorm(sample["nodule"][:, :, :, config["cube_voxelsize"] // 2])
        img_01 = norm2(img)
        # img = norm(sample["nodule"][:, :, :, config["cube_voxelsize"] // 2]).to(torch.uint8)
        # scan = norm2(scan).astype(np.uint8)

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        im0 = ax[0, 0].imshow(img.numpy()[0], cmap="gray")
        ax[0, 0].axis('off')
        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im0, cax=cax, orientation='vertical')

        im1 = ax[0, 1].imshow(clip_scan[:, :, clip_scan.shape[2] // 2], cmap="gray")
        ax[0, 1].axis('off')
        divider = make_axes_locatable(ax[0, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        im2 = ax[1, 0].imshow(img_01.numpy()[0], cmap="gray")
        ax[1, 0].axis('off')
        divider = make_axes_locatable(ax[1, 0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        im3 = ax[1, 1].imshow(scan[:, :, scan.shape[2] // 2], cmap="gray")
        ax[1, 1].axis('off')
        divider = make_axes_locatable(ax[1, 1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im3, cax=cax, orientation='vertical')

        fig.suptitle(patient_id)
        fig.tight_layout()
        writer.add_figure("sample_fig", fig, i)

    print("Done")
