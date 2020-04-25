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

    # After resampling spacing becomes 1mm / 1 px
    nodulecube = ndimage.zoom(nodulecube, spacing, order=5, prefilter=False)
    sh = nodulecube.shape
    resampled_cube = ndimage.zoom(
        nodulecube,
        (cube_voxelsize / sh[0], cube_voxelsize / sh[1], cube_voxelsize / sh[2]),
        order=5,
        prefilter=False,
    )

    return resampled_cube
