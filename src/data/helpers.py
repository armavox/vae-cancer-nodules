import logging
import warnings
from typing import Union

import numpy as np
from scipy import ndimage


logger = logging.getLogger("data.helpers")
warnings.filterwarnings("ignore", ".*output shape of zoom.*")


def extract_cube(
    series_volume: np.ndarray,
    spacing: Union[np.ndarray, list],
    nodule_coords: np.ndarray,
    cube_voxelsize: int = 48,
    extract_size_mm: float = 42.0,
) -> np.ndarray:
    """Extracts from series at image coordinates xyz the cube
    of size cube_size_mm^3 mm. Resampled to voxelsize of cube_size^3 voxels.

    Parameters
    ----------
    series_volume : np.ndarray
        Series image volume. Axis: [x, y, z]
    spacing : Union[np.ndarray, list]
        List of (x_pixel_spacing, y_pixel_spacing, z_slice_spacing)
    nodule_coords : np.ndarray
        Coords of the nodule centroid in the image frame (voxel) [x, y, z].
    cube_voxelsize : int, optional
        Voxel size of the extracted cube to resample. Should be larger than cube_size_mm, by default 48.
    cube_size_mm : int, optional
        Size in millimeters of the extracted cube, by default 42

    Returns
    -------
    np.ndarray
        Volume with the extracted nodule of cube_size_mm size anf cube_voxelsize shape and with
        pixel spacing in all dimensions 1 mm / 1 px.
"""

    xyz = nodule_coords.astype("int")  # xyz_coords[::-1].astype("int")
    halfcube_voxelsize = (extract_size_mm / spacing / 2).astype("int")

    # Check if padding is necessary
    if np.any(xyz < halfcube_voxelsize) or np.any(xyz + halfcube_voxelsize > series_volume.shape):
        maxsize = max(halfcube_voxelsize)
        series_volume = np.pad(series_volume, pad_width=maxsize, mode="constant", constant_values=-1000)
        xyz = xyz + maxsize

    # Extract cube from series at xyz. Scan is [z, y, x]
    nodulecube = series_volume[
        xyz[0] - halfcube_voxelsize[0] : xyz[0] + halfcube_voxelsize[0],
        xyz[1] - halfcube_voxelsize[1] : xyz[1] + halfcube_voxelsize[1],
        xyz[2] - halfcube_voxelsize[2] : xyz[2] + halfcube_voxelsize[2],
    ]

    # After resampling spacing becomes 1mm / 1 px. Now fine-tune resample up to cube_voxelsize
    to_spacing = spacing * (cube_voxelsize / np.round(nodulecube.shape * spacing).astype("int"))
    resampled_cube = ndimage.zoom(
        nodulecube,
        to_spacing,  # (cube_voxelsize / sh[0], cube_voxelsize / sh[1], cube_voxelsize / sh[2]),
        order=5,
        prefilter=False,
    )
    # fig, ax = plt.subplots(1, 3, figsize=(8, 12))
    # [axi.set_axis_off() for axi in ax.ravel()]
    # gs = plt.GridSpec(4, 3, figure=fig)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax1 = fig.add_subplot(gs[0, 1])
    # ax2 = fig.add_subplot(gs[0, 2])
    # ax3 = fig.add_subplot(gs[1:, 0:])

    # # ax0.imshow(nodulecube_orig[:, :, nodulecube_orig.shape[2]//2], cmap="gray")
    # ax1.imshow(nodulecube[:, nodulecube.shape[1]//2, :], cmap="gray")
    # ax2.imshow(resampled_cube[:, resampled_cube.shape[1]//2, :], cmap="gray")
    # ax3.imshow(series_volume[:, :, int(nodule_coords[2])], cmap="gray")
    # plt.savefig('abc2.png')
    return resampled_cube
