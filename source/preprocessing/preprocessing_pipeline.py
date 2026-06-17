# source/preprocessing/run_pipeline.py
"""
Functions to run the full preprocessing pipeline. Loop through all voltages, load friction, correlation function,
and adiabatic force, and perform the ESPRIT decomposition on the friction and correlation function,
before saving everything to the appropriate results directory.
    
USAGE:
-----
from repo directory, run python3 -m scripts.run_preprocessing
"""

import numpy as np
from typing import (
    Dict, 
    List, 
    Tuple,
    Optional
)
from numpy.typing import NDArray # type: ignore
import logging
from pathlib import Path
from source.preprocessing import (
    pole_management,
    coordinate_grid,
    pole_dataclass
)
from source.preprocessing.esprit_decomposition import esprit_high_level
from source.preprocessing.lorentzian_decomposition import lorentzian_high_level
from source.utils.enums import FunctionType , ForcesType , ElObsType
from source.utils import data_loader, file_walker
from source.utils.io_utils import ensure_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # default logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def decompose_single_x_coordinate(
        xdir: Path,
        cfg: dict,
        function_type: FunctionType,
        prev_poles: Optional[List[pole_dataclass.Pole]],
        center_poles: Optional[List[pole_dataclass.Pole]]
) -> List[pole_dataclass.Pole]:
    """
    Decompose a single x-coordinate function into poles using ESPRIT.

    INPUT
    ----------
    xdir: Path
        Directory containing the function data for this x-coordinate
    cfg: dict
        Configuration dictionary with decomposition parameters
    function_type : FunctionType
        Specifies whether to decompose FRICTION or CORRFUNC
    initial_guess : List of Pole
        Initial guess from previous x-value to put into Lorentzian decomposition

    OUTPUT
    ------
    poles: List of Pole
        List of Pole obects from the ESPRIT decomposition
    
    Note: For FRICTION, the negative of the integrand is taken before decomposition
    """
    
    input_poles,time_vec = esprit_high_level.generate_decomposition(
        cfg,function_type,xdir,initial_guess=prev_poles
    )

    if function_type is FunctionType.CORRFUNC:
        if prev_poles is not None and center_poles is not None:
            final_poles = lorentzian_high_level.lorentzian_decomposition_w_input(
                cfg,input_poles,prev_poles,time_vec,center_poles,function_type
            ) 
        else: 
            final_poles = lorentzian_high_level.lorentzian_decomposition(
                cfg,input_poles,time_vec
            )
    elif function_type is FunctionType.FRICTION:
        # if prev_poles is not None:
        #     input_poles = pole_management.track_poles_pair(prev_poles,input_poles)
        # if prev_poles is not None and center_poles is not None:
        #     # final_poles = esprit_high_level.esprit_smoothing(
        #     #     cfg,function_type,xdir,input_poles=input_poles,prev_poles=prev_poles
        #     # )
        #     final_poles = lorentzian_high_level.lorentzian_decomposition_w_input(
        #         cfg,input_poles,prev_poles,time_vec,center_poles,function_type
        #     )
        # else: 
        final_poles = input_poles

    return final_poles

def process_single_x_coordinate(
    itrx,
    x_ind_prev,
    x_max_ind,
    markovian_forces_list,
    elobs_list,
    xdir,
    poles_per_type,
    cfg
):
    
    for forcestype in ForcesType:
        markovian_forces_list[forcestype][itrx] = data_loader.load_markovian_force(xdir,forcestype)
    for elobstype in ElObsType:
        elobs_list[elobstype][itrx] = data_loader.load_markovian_el_obs(xdir,elobstype)
    for ft in FunctionType:
        prev_poles = poles_per_type[ft][x_ind_prev] if x_ind_prev is not None else None
        prev_poles_list = [p for p in prev_poles if p is not None] if prev_poles is not None else None
        center_poles = poles_per_type[ft][x_max_ind]
        center_poles_list = [p for p in center_poles if p is not None] if center_poles is not None else None
        pole_output = decompose_single_x_coordinate(
            xdir, cfg, ft, prev_poles=prev_poles_list , center_poles=center_poles_list
            )
        if prev_poles is not None:
            pole_output = pole_management.enforce_conjugate_weights(pole_output)
            pole_output = pole_management.track_poles_pair(prev_poles,pole_output)
        poles_per_type[ft][itrx] = pole_output

    return markovian_forces_list,elobs_list,poles_per_type

def load_and_decompose_single_voltage(
        voltage_dir: Path,
        cfg: dict
) -> Tuple[
    NDArray[np.floating],
    Dict[FunctionType, List[List[pole_dataclass.Pole]]],
    Dict[ForcesType, NDArray[np.floating]],
    Dict[ElObsType, NDArray[np.floating]]
]:
    """
    Process all x-coordinates with decompose_single_x_coordinate for a single voltage
    
    INPUT
    -----
    voltage_dir: Path
        Directory containing all x-coordinates for this voltage
    cfg: dict
        Configuration dictionary with decomposition parameters

    OUTPUT
    ------
    x_coordinate: list of floats
        List of x-coordinates for this voltage
    poles_per_type: list of list of Pole
        List of list of Pole obects from the ESPRIT decomposition for this voltage
    adiabatic_force_arr: list of floats
        Lit of adiabatic force values at each x-coordinate for this voltage
    """
    x_max_ind,x_coordinate_arr = coordinate_grid.x_grid_finder(voltage_dir,cfg)
    len_x_coordinate_arr = len(x_coordinate_arr)
    markovian_forces_list: Dict[ForcesType, np.ndarray] = {
        forcestype: np.zeros(len_x_coordinate_arr, dtype=float) for forcestype in ForcesType
    }
    poles_per_type: Dict[FunctionType, List[List[pole_dataclass.Pole]]] = {
    ft: [[] for _ in range(len_x_coordinate_arr)] for ft in FunctionType
    }
    elobs_list: Dict[ElObsType, np.ndarray] = {
        elobstype: np.zeros(len_x_coordinate_arr, dtype=float) for elobstype in ElObsType
    }

    xdir = file_walker.add_xdir_to_path_given_x(x_coordinate_arr[x_max_ind],voltage_dir)
    for forcestype in ForcesType:
        markovian_forces_list[forcestype][x_max_ind] = data_loader.load_markovian_force(xdir,forcestype)
    for ft in FunctionType:
        pole_output = decompose_single_x_coordinate(xdir, cfg, ft, prev_poles=None,center_poles=None)
        poles_per_type[ft][x_max_ind] = pole_output
    
    logger.info(f"Starting preprocessing for voltage {voltage_dir.name}, center x = {x_coordinate_arr[x_max_ind]}")
    sweeps = [
        range(x_max_ind + 1, len_x_coordinate_arr),
        range(x_max_ind - 1, -1, -1)
    ]
    sweep_direction_arr = ["single","single"]
    processed_coordinates = [False] * len_x_coordinate_arr
    for sweep,sweep_dir in zip(sweeps,sweep_direction_arr):
        for x_ind in sweep:
            x_ind_prev = coordinate_grid.generate_prev_x_ind(x_ind,x_max_ind,sweep_dir)
            x = x_coordinate_arr[x_ind]
            print(x)
            xdir = file_walker.add_xdir_to_path_given_x(x_coordinate_arr[x_ind],voltage_dir)
            logger.debug(f"Processing x-coordinate {x} (index {x_ind})")
            markovian_forces_list,elobs_list,poles_per_type = process_single_x_coordinate(
                x_ind,x_ind_prev,x_max_ind,markovian_forces_list,
                elobs_list,xdir,poles_per_type,cfg
                )
            processed_coordinates[x_ind] = True
                
    # Redo initial x-value to ensure smoothness
    for ft in FunctionType:
        x_ind_prev = x_max_ind + 1
        xdir = file_walker.add_xdir_to_path_given_x(x_coordinate_arr[x_max_ind],voltage_dir)
        markovian_forces_list,elobs_list,poles_per_type = process_single_x_coordinate(
                x_max_ind,x_ind_prev,x_max_ind,markovian_forces_list,
                elobs_list,xdir,poles_per_type,cfg
                )
        poles_per_type[ft][x_max_ind] = pole_output
    
    logger.info(f"Voltage {voltage_dir.name} preprocessing completed.")
    
    return x_coordinate_arr,poles_per_type,markovian_forces_list,elobs_list

def get_voltage_results_dir(
        voltage_dir: Path,
        cfg: dict
) -> Path:
    """
    Return the full path where results for this voltage should be stored.
    
    INPUT
    -----
    voltage_dir: Path
        Directory containing all x-coordinates for this voltage
    cfg: dict
        Configuration dictionary with decomposition parameters

    OUTPUT
    ------
    full_save_path: Path
        Absolute path where the results for this voltage should be stored.
    """
    results_root = cfg["results_root"]
    el_forces_root = cfg["el_forces_root"]
    data_root = cfg["data_root"]
    rel_path = voltage_dir.relative_to(data_root)
    
    full_save_path = results_root / rel_path / el_forces_root
    
    return ensure_directory(full_save_path)

def save_results(
        save_dir: Path,
        cfg: dict,
        name: str,
        **results_to_save: NDArray[np.floating],
) -> None:
    """
    Save one or more named arrays to a .npz file
    
    INPUT
    -----
    save_dir: Path
        Save directory for this file
    cfg: dict
        Configuration dictionary with decomposition parameters
    name: string
        Name of save file
    resutls_to_save: NDArray of floats
        Data arrays to be saved 
    """
    el_forces_save_dir = get_voltage_results_dir(save_dir,cfg) / f"{name}.npz"
    np.savez(el_forces_save_dir, **results_to_save)

def process_single_voltage(voltage_dir,cfg):

    x_coordinate_arr,poles_per_type,markovian_forces_list,elobs_list = load_and_decompose_single_voltage(voltage_dir,cfg)
    force_kwargs = {
    force.tag: markovian_forces_list[force]
    for force in ForcesType
    }
    save_results(
        save_dir=voltage_dir,
        cfg=cfg,
        name=f"unprocessed_markovian_forces",
        x=x_coordinate_arr,
        **force_kwargs
    )
    elobs_kwargs = {
    elobs.tag: elobs_list[elobs]
    for elobs in ElObsType
    }
    save_results(
        save_dir=voltage_dir,
        cfg=cfg,
        name=f"unprocessed_el_observables",
        x=x_coordinate_arr,
        **elobs_kwargs
    )
    for ft in FunctionType:
        poles_arr = pole_management.poles_to_arrays(x_coordinate_arr, poles_per_type[ft])
        output_name = f"unprocessed_weights_frequencies_{ft.tag}"
        save_results(
            voltage_dir,
            cfg,
            output_name,
            x=poles_arr["x"],
            freq_real=poles_arr["freq_real"],
            freq_imag=poles_arr["freq_imag"],
            weight_real=poles_arr["weight_real"],
            weight_imag=poles_arr["weight_imag"]
        )

    logger.info(f"{voltage_dir.relative_to(cfg['raw_data_root'])} processing completed")
    logger.info(f"Number of x-coordinates found: {len(x_coordinate_arr)}")

def run_preprocessing_pipeline(
        cfg:dict
) -> None:
    """
    Run the full preprocessing timeline. Loop through all voltages, load friction, correlation function,
    and adiabatic force, and perform the ESPRIT decomposition on the friction and correlation function,
    before saving everything to the appropriate results directory.
    
    INPUT
    -----
    cfg: dict
        Configuration dictionary with decomposition parameters
    """
    data_root = Path(cfg["raw_data_root"])
    for voltage_dir in file_walker.iter_voltage_dirs(data_root,cfg):
        process_single_voltage(voltage_dir,cfg)
        

if __name__ == "__main__":

    from source.utils.config import load_config 

    cfg = load_config()
    # run_full_pipeline(cfg)
