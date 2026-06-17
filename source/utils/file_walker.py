# source/preprocessing/file_walker.py
"""
Searches the current data file for subdirectories of 
x_coordinates to be loaded

Provides:
- list of voltage directories to loop through, given voltage range
- list of x_coordinate directories within each voltage directory, found
  by exploration
- x_coordinate associated with each x_coordinate directory
"""

from pathlib import Path
import numpy as np

def generate_voltage_dir(data_root,voltage):

    return data_root / Path(f"voltage_{voltage}eV")

def iter_voltage_dirs(data_root,cfg):
    
    min_voltage = cfg["voltage"]["min"]
    max_voltage = cfg["voltage"]["max"]
    step_voltage = cfg["voltage"]["step"]

    for voltage in np.arange(min_voltage,max_voltage+step_voltage,step_voltage):
        yield generate_voltage_dir(data_root,voltage)

def add_xdir_to_path_given_x(x,parent):

    xdir = x_dir_given_x_coordinate(x)
    total_path = Path(parent) / xdir
    
    return total_path

def x_dir_given_x_coordinate(x):

    """
    Convert a float x into directory name:
      x >= 1     -> x_10.2
      x <= -1    -> x_-10.2
      0 < x < 1  -> x_.2
      -1 < x < 0 -> x_-.2
    """
    if x >= 1:
        s =f"{x:.1f}"
    elif x <= -1:
        s =f"{x:.1f}"
    elif x > 0:
        s =f"{x:.1f}".replace("0.",".")  # 0.2 -> .2
    else:  # -1 < x < 0
        s = f"{x:.1f}".replace("-0.","-.")  # -0.2 -> -_.2

    return Path(f"x_{s}")

def find_x_coordinate_dirs(voltage_dir):
    """
    Yield all x-coordinate directories inside a given voltage directory.

    Inputs
    ------
    voltage_dir : Path or str
        Path to a voltage volder, e.g. 'data/voltage_0V'

    Outputs
    ------
    Path
        Path to each x-coordinate subdirectory
    """
    voltage_dir = Path(voltage_dir)

    # Iterate over all entries in voltage_dir
    for entry in voltage_dir.iterdir():
        if entry.is_dir():
            yield entry


if __name__ == "__main__":
    
    from source.utils.config import load_config
    cfg = load_config()
    data_root = cfg["data_identifier_root"]
    print(data_root)
    for voltage_dir in iter_voltage_dirs(data_root,cfg):
        print(f"Voltage dir: {voltage_dir}")
        for xdir in find_x_coordinate_dirs(voltage_dir):
            print(f"x dir: {xdir}")
            