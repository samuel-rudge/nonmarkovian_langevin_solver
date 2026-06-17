# source/langevin_propagation/load_forces.py

import numpy as np
from typing import (
    Dict, 
    List, 
    Tuple,
    Callable
)
from numpy.typing import NDArray #type: ignore
from pathlib import Path
from scipy.interpolate import CubicSpline
from source.utils.enums import (
    FunctionType,
    PoleType,
    RealImag,
    ForcesType,
    ElObsType
)
import logging
from matplotlib import pyplot as plt

def load_force_array(forces_dir,filename):

    file_path = forces_dir / filename
    try:
        data_array = np.load(file_path)
    except (IndexError, ValueError):
        raise ValueError(f"File missing or malformed: {file_path}")

    return data_array

def interpolate_electronic(data_arr,tag):

    return (CubicSpline(
        data_arr["x"],
        data_arr[tag]
        )
    )

def interpolate_adiabatic_force(data_arr):
    
    return (CubicSpline(
        data_arr["x"],
        data_arr["Adiabatic_Force"]
        )
    )

def interpolate_markovian_friction(data_arr):
    
    return (CubicSpline(
        data_arr["x"],
        data_arr["Friction"]
        )
    )

def interpolate_markovian_corrfunc(data_arr):
    
    return (CubicSpline(
        data_arr["x"],
        data_arr["Corrfunc"]
        )
    )

class NearestValue:
    """Return value at the nearest x-coordinate from a discrete set."""
    def __init__(self, x_vec, y_vec):
        self.x_vec = np.array(x_vec)
        self.y_vec = np.array(y_vec)

    def __call__(self, x):
        x = np.atleast_1d(x)
        # Find indices of nearest x
        idx = np.abs(self.x_vec[None, :] - x[:, None]).argmin(axis=1)
        return self.y_vec[idx] if len(x) > 1 else self.y_vec[idx[0]]

def interpolate_poles(x_coordinate_vec,data_arr,ft):

    n_terms = data_arr.shape[1]
    if ft is FunctionType.FRICTION:
        return [CubicSpline(x_coordinate_vec,data_arr[:,itrp]) for itrp in range(n_terms)]
    elif ft is FunctionType.CORRFUNC:
        return [NearestValue(x_coordinate_vec, data_arr[:, i]) for i in range(n_terms)]

# def interpolate_poles(x_coordinate_vec,data_arr,ft):

#     n_terms = data_arr.shape[1]
#     return [CubicSpline(x_coordinate_vec,data_arr[:,itrp]) for itrp in range(n_terms)]

def generate_grid_limits(x):

    return np.array([np.min(x),np.max(x)])

def load_and_interpolate_el_obs(cfg,voltage):

    results_root = cfg["results_root"]
    el_forces_root = cfg["el_forces_root"]
    processed_el_forces_dir = results_root / cfg["system_identifier_dir"] / f"voltage_{voltage:.2f}eV" / el_forces_root
    el_obs_arr = load_force_array(processed_el_forces_dir,"unprocessed_el_observables.npz")
    interp_el_obs_arr = []
    interp_el_obs_arr.append(interpolate_electronic(el_obs_arr,
                                                    ElObsType.AD_CURRENT.tag))
    
    return interp_el_obs_arr

def load_and_interpolate_forces(cfg,voltage):

    results_root = cfg["results_root"]
    el_forces_root = cfg["el_forces_root"]
    processed_el_forces_dir = results_root / cfg["system_identifier_dir"] / f"voltage_{voltage:.2f}eV" / el_forces_root
    markovian_value = cfg["simulation"]["markovian"]
    interp_forces_arr = []
    # Do adiabatic force
    markovian_forces_arr = load_force_array(processed_el_forces_dir,"unprocessed_markovian_forces.npz")
    interp_forces_arr.append(interpolate_electronic(markovian_forces_arr,
                                                    ForcesType.AD_FORCE.tag))
    grid_limits = generate_grid_limits(markovian_forces_arr["x"])
    # Do friction and correlation function pole decompositions
    if markovian_value:
        interp_forces_arr.append(interpolate_electronic(markovian_forces_arr,
                                                    ForcesType.FRICTION.tag))
        interp_forces_arr.append(interpolate_electronic(markovian_forces_arr,
                                                    ForcesType.CORRFUNC.tag))
    else:
        for ft in FunctionType:
            func_arr = load_force_array(processed_el_forces_dir,f"unprocessed_weights_frequencies_{ft.tag}.npz")
            x_coordinate_vec = func_arr["x"]
            interp_func_arr = []
            for pt in PoleType:
                interp_pole_arr = []
                for ri in RealImag:
                    interp_pole_arr.append(
                        interpolate_poles(x_coordinate_vec,func_arr[f"{pt.tag}_{ri.tag}"],ft)
                        )
                interp_func_arr.append(interp_pole_arr)
            interp_forces_arr.append(interp_func_arr)

    return interp_forces_arr,grid_limits

if __name__ == "__main__":

    import numpy as np
    from source.utils.config import load_config
    from matplotlib import pyplot as plt

    cfg = load_config()
    interp_forces_arr,grid_limits = load_and_interpolate_forces(cfg,0.0)
    x = np.linspace(-25,25,100)
    af = interp_forces_arr[2][1][0][5](x)
    plt.plot(x,af) ; plt.show()
