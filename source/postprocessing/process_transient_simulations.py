# source/postprocessing/process_transient_simulations.py

import numpy as np
from pathlib import Path
from source.langevin_propagation import (
    prepare_forces,
    trajectory_sampling
)
from source.postprocessing import results_io
from source.utils.results_layout import ResultsLayout
import re

def initialize_ensemble_av_data(simulation_cfg, n_timesteps, n_vib):

    data_storage_cfg = simulation_cfg["data_storage"]
    dt = simulation_cfg["dt"]
    ensemble_av_data = {
        "time_vec": np.linspace(0,n_timesteps*dt,n_timesteps),
        "physical_state": np.zeros((n_timesteps,n_vib,2), dtype=np.float64)
    }
    if data_storage_cfg.get("store_energy",False):
        ensemble_av_data["kinetic_energy"] = np.zeros((n_timesteps,n_vib), dtype=np.float64)
        ensemble_av_data["potential_energy"] = np.zeros((n_timesteps,n_vib), dtype=np.float64)
    if data_storage_cfg.get("store_current",False):
        ensemble_av_data["current"] = np.zeros((n_timesteps,n_vib), dtype=np.float64)
    if data_storage_cfg.get("store_stoch_force",False):
        ensemble_av_data["stoch_force"] = np.zeros((n_timesteps), dtype=np.float64)
        ensemble_av_data["corrfunc_exact"] = np.zeros((n_timesteps), dtype=np.float64)
    if data_storage_cfg.get("store_current",False):
        ensemble_av_data["adiabatic_current"] = np.zeros(n_timesteps,dtype=np.float64)

    return ensemble_av_data

def generate_corrfunc(x,time_vec,interp_stoch_arr):

    corrfunc = np.zeros(len(time_vec),dtype=np.float64)
    n_terms = len(interp_stoch_arr[0][0])
    for itr_pole in np.arange(n_terms):
        freq_real = interp_stoch_arr[0][0][itr_pole](x)
        freq_imag = interp_stoch_arr[0][1][itr_pole](x)
        weights_real = interp_stoch_arr[1][0][itr_pole](x)
        corrfunc += weights_real * np.exp(-freq_real * time_vec) * np.cos(freq_imag * time_vec)

    return corrfunc

def get_transient_trajectory_indices(base_dir):
    """
    Scan base_dir for files named traj_transient_<int>.<ext>
    and return a sorted list of indices.
    """
    pattern = re.compile(r"traj_transient_(\d+)\.")

    indices = []

    for path in Path(base_dir).iterdir():
        match = pattern.search(path.name)
        if match:
            indices.append(int(match.group(1)))

    return sorted(indices)

def calculate_ensemble_averages(cfg,voltage,traj_list=None,traj_type="transient"):

    results_layout = ResultsLayout(cfg,voltage)
    results_writer = results_io.ResultsWriter(results_layout,cfg)
    simulation_cfg = cfg["simulation"]
    if traj_type == "transient":
        n_timesteps = simulation_cfg["n_timesteps"]
    elif traj_type == "ss":
        n_timesteps = trajectory_sampling.ss_n_timesteps(cfg)
    n_vib = cfg["n_vib"]
    interp_forces_arr,grid_limits = prepare_forces.load_and_interpolate_forces(cfg,voltage)
    ensemble_av_data = initialize_ensemble_av_data(simulation_cfg, n_timesteps, n_vib)
    if traj_list is None:
        n_traj = simulation_cfg["n_traj"]
        #traj_list = np.arange(n_traj)
        traj_list = get_transient_trajectory_indices(results_layout.transient_traj)
    n_traj = len(traj_list)
    for traj_id in traj_list:
        traj_data = results_writer.load_transient(traj_id)
        for key in traj_data:
            if key != "time_vec":
                if key == "stoch_force":
                    forces = traj_data[key] - traj_data[key].mean()
                    temp_data = np.correlate(forces,forces,mode="full")
                    temp_data = temp_data[temp_data.size // 2:]
                    temp_data /= np.arange(len(temp_data),0,-1)
                    ensemble_av_data[key] += temp_data 
                    ensemble_av_data["corrfunc_exact"] += generate_corrfunc(
                        traj_data["physical_state"][0][0,0],
                        ensemble_av_data["time_vec"],
                        interp_forces_arr[2]
                        )
                elif key == "aux_stoch":
                    forces = traj_data[key] - traj_data[key].mean()
                    ensemble_av_data[key] += forces ** 2
                else:
                    ensemble_av_data[key] += traj_data[key]
        traj_data.close()

    for key in ensemble_av_data:
        if key != "time_vec":
            ensemble_av_data[key] /= n_traj
    
    results_writer.save_ensemble_av_data(ensemble_av_data)

if __name__ == "__main__":

    from source.utils.config import load_config
    import matplotlib.pyplot as plt
    cfg = load_config()

    voltage = cfg["voltage"]["min"]
    # results_dir = Path(cfg["results_transient_root"] / cfg["system_identifier_dir"] / f"voltage_{voltage:.2f}eV")
    
    ensemble_av_data = calculate_ensemble_averages(cfg,voltage)
    # plt.plot(ensemble_av_data["time_vec"],ensemble_av_data["physical_state"][:,0,0])
    # plt.show()