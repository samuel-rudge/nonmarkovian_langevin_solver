# source/preprocessing/coordinate_grid.py
"""
Creates x-grid for given voltage and finds "center of grid"; point at which 
correlation function and/or friction is maximum.

Provides:
---------
- generate_x_grid_for_single_voltage(voltage_dir,cfg)
    Given voltage and path, finds all x-folders within voltage and collects
    them into grid of x-coordinates

- sweep_from_center(x_max_ind,len_x_grid,x_grid):
    Defines loop going out from center of x-grid

- generate_prev_x_ind(itrx,x_max_ind):
    Generates previous x index (-1 if we are beyond center and +1 if before)

Usage:
------
from source.preprocessing import coordinate_grid
x_max_ind,x_coordinate_grid_vec = coordinate_grid.x_grid_finder(voltage_dir,cfg)

Notes:
------
"""

from source.utils import file_walker,data_loader
from source.utils.enums import ForcesType
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Optional
from numpy.typing import NDArray

def x_grid_finder(
        voltage_dir: Path,
        cfg: dict
) -> Tuple[int, NDArray[np.floating]]:

    """
    Build x-cooridnate grid for given voltage directory and determine the index
    of the grid center

    Parameters
    ----------
    voltage_dir: Path
        Directory containing all x-coordinate subdirectories for a given voltage
    cfg: dict
        Configuration dictionary (must contain 'grid limit')
    
    Returns
    -------
    x_max_ind: int
        Index of the center x-coordinate in filtered grid (can set to center index 
        or index of maximum value of correlation function)
    x_coordinate_grid_vec: NDAarray[np.floating]
        Sorted and filtered array of x-coordinates
    """
    x_coordinate_list: list[float] = []
    grid_limit: float = cfg["grid_limit"]
    center_ind_type: str = cfg["decomposition"]["center_index_type"]
    if center_ind_type == "max":
        markovian_corrfunc_list: list[float] = []   

    for xdir in file_walker.find_x_coordinate_dirs(voltage_dir):
        x_value = data_loader.extract_x_coordinate_from_dir(xdir)
        x_coordinate_list.append(x_value)
        if center_ind_type == "max":
            markovian_corrfunc = data_loader.load_markovian_force(xdir,ForcesType.CORRFUNC)
            markovian_corrfunc_list.append(markovian_corrfunc)
        
    x_coordinate_arr = np.asarray(x_coordinate_list, dtype=float)
    ordered_x = np.argsort(x_coordinate_arr)
    x_coordinate_arr = x_coordinate_arr[ordered_x]
    grid_mask = (x_coordinate_arr <= grid_limit) & (x_coordinate_arr >= -grid_limit)
    if center_ind_type == "max":
        markovian_corrfunc_arr = np.asarray(markovian_corrfunc_list, dtype=float)
        markovian_corrfunc_arr = np.asarray(markovian_corrfunc_arr, dtype=float)[ordered_x]    
        x_max_ind = int(np.argmax(markovian_corrfunc_arr[grid_mask]))
    elif center_ind_type == "center":
        x_max_ind = len(x_coordinate_arr[grid_mask]) // 2
    
    x_coordinate_grid_arr = x_coordinate_arr[grid_mask]
    
    return x_max_ind,x_coordinate_grid_arr

def generate_next_x_ind(
        x_ind,
        len_x_coordinate_arr,
        sweep_dir
):
    
    if sweep_dir == "bi":
        if x_ind + 1 < len_x_coordinate_arr:
            x_ind_next = x_ind + 1
        else:
            x_ind_next = None
    else:
        x_ind_next = None

    return x_ind_next

def sweep_from_center(
        x_max_ind: int,
        x_grid: NDArray[np.floating]
) -> Iterator[Tuple[int, float]]:
    """
    Generator that yields tuples of x-indices and values, starting from the center of the grid 
    and sweeping outwards, first increasing then decreasing.

    Args:
        x_max_ind: Index of center point in x_grid
        x_grid: 1D array of x-coordinates
    
    Yields:
        Tuple of (index, x-value)
    """
    len_x_grid = len(x_grid)

    # sweep up
    for i in range(x_max_ind+1,len_x_grid):
        yield i,x_grid[i]
    
    # sweep down
    for i in range(x_max_ind-1,-1,-1):
        yield i,x_grid[i]

def generate_prev_x_ind(
        itrx: int,
        x_max_ind: int,
        sweep_dir: str
) -> Optional[int]:
    """
    Generate the "previous" x index relative to the center index (x_max_ind).

    Returns:
        - int: index of the previous x-coordinate to use as initial guess
        - None: if current index is the center
    """
    
    if sweep_dir == "single":
        if itrx < x_max_ind:
            return int(itrx + 1)

        if itrx > x_max_ind:
            return int(itrx - 1)
        
        if itrx == x_max_ind:
            return None
    elif sweep_dir == "bi":
        if itrx - 1 > -1:
            return int(itrx - 1)
        else:
            return None
