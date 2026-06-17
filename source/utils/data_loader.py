# source/preprocessing/data_loader.npy
"""
Loads time-dependent functions (friction or correlation function) from the 
data folder hierarchy and extracts x-coordinate information from folder names. 

Provides:
---------
- load_function_to_decompose(function_type,function_path)
    Given the function_type ("friction" or "corrfunc") and the location of the function 
    (function_path), it returns the time_vector and the friction or corrfunc integrand as 
    single data array, and also returns the x-coordinate as a float

- extract_x_coordinate_from_dir(x_dir):
    Extract the x-coordinate float # from a folder name like '.../x_#'

Usage:
------
from source.preprocessing import data_loader
data,x = data_loader.load_function_to_decompose("friction",x_dir)

Notes:
------
- The folder structure is expected to be:
    data_root/
        molecule_lead_coupling_temperature_root/
            delta_root/
                tier_root/
                    voltage_root_<voltage_value>/
                        x_<x_value>/
                            friction_integrand_heom.dat
                            corrfunc_integrand_heom.dat

- This module raises ValueError for invalid function_type or malformed folder names.

- FileNotFoundError is raised if the expected data file does not exist.
"""

from pathlib import Path
import numpy as np
from source.utils.enums import FunctionType , ForcesType , ElObsType


def load_function_to_decompose(
        function_type: FunctionType,
        function_path: Path
):
    """
    Loads a time-dependent function given a function path

    INPUTS
    ------
    function_type : str 
        Can only take values of "friction" or "corrfunction"

    function_path : Path or str
        Path to folder where function is stored

    OUTPUTS
    ------
    Path
        Path to each x-coordinate subdirectory
    """
    x_value = extract_x_coordinate_from_dir(function_path)
    function_path = Path(function_path)
    
    try:
        file_path = function_path / function_type.filename
    except KeyError:
        raise ValueError(f"Unknown FunctionType: {function_type}")

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Load N_timesteps x 2 array with time in first column and function integrand in second column
    try:
        data_array = np.genfromtxt(file_path)
    except (IndexError, ValueError):
        raise ValueError(f"File missing or malformed: {file_path}")
    
    if data_array.shape[1] < 2:
        raise ValueError(f"Data file {file_path} must have at least two columns")

    return data_array[:,0],data_array[:,1],x_value

def load_markovian_el_obs(
        function_path: Path,
        elobs_type: ElObsType
):
    
    file_path = Path(function_path) / Path(elobs_type.elobs_filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load N_x x 2 array with x in first column and force in second column
    try:
        data_array = np.genfromtxt(file_path,names=["Coord_Val",elobs_type.tag],skip_header=1)
    except (IndexError, ValueError):
        raise ValueError(f"File missing or malformed: {file_path}")
    
    return data_array[elobs_type.tag]

def load_markovian_force(
        function_path: Path,
        force_type: ForcesType
):
    
    file_path = Path(function_path) / Path(force_type.markovian_filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load N_x x 2 array with x in first column and force in second column
    try:
        data_array = np.genfromtxt(file_path,names=["Coord_Val",force_type.tag],skip_header=1)
    except (IndexError, ValueError):
        raise ValueError(f"File missing or malformed: {file_path}")
    
    return data_array[force_type.tag]

def extract_x_coordinate_from_dir(x_coordinate_dir):

    folder_name = x_coordinate_dir.name
    try:
        #Supports decimals and signed numbers
        x_value = float(folder_name.split("x_")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Cannot extract x-coordinate from folder name: {folder_name}")
    
    return x_value

if __name__ == "__main__":

    from source.utils import file_walker
    from source.utils.config import load_config

    cfg = load_config()
    data_root = cfg["data_root"]
    for voltage_dir in file_walker.iter_voltage_dirs(data_root,cfg):
        print(f"Voltage dir: {voltage_dir}")
        for xdir in file_walker.find_x_coordinate_dirs(voltage_dir):
            print(f"x dir: {xdir}")
            time_vec,function_integrand,x_coordinate = \
                load_function_to_decompose(FunctionType.FRICTION,xdir)
