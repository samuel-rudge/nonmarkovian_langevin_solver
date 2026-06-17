# source/langevin_propagation/run_trajectories.py

import numpy as np
from source.langevin_propagation.integrator_aboba import propagate_aboba_onestep,propagate_aboba_markovian_onestep
from source.langevin_propagation import (
    prepare_forces,
    prepare_initial_conditions,
    equations_of_motion
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from source.utils.io_utils import ensure_directory
from source.langevin_propagation.integrator_aboba import generate_stoch_force

def initialize_traj_data(data_storage_cfg, n_timesteps, n_vib):

    traj_data = {
        "physical_state": np.zeros((n_timesteps,n_vib,2), dtype=float)
    }
    if data_storage_cfg.get("store_energy",False):
        traj_data["kinetic_energy"] = np.zeros((n_timesteps,n_vib), dtype=float)
        traj_data["potential_energy"] = np.zeros((n_timesteps,n_vib), dtype=float)
    if data_storage_cfg.get("store_current",False):
        traj_data["adiabatic_current"] = np.zeros((n_timesteps), dtype=float)
    if data_storage_cfg.get("store_stoch_force",False):
        traj_data["stoch_force"] = np.zeros((n_timesteps), dtype=float)

    return traj_data

def store_traj_data(traj_data,itrt,physical_state,stoch_force,cfg,
                    adiabatic_current=None):

    traj_data["physical_state"][itrt,:] = physical_state
    data_storage_cfg = cfg["simulation"]["data_storage"]
    if data_storage_cfg.get("store_energy",False):
        vib_freq = cfg["vib_freq"]
        kinetic_energy = 0.5 * vib_freq * physical_state[0,1]**2
        traj_data["kinetic_energy"][itrt,:] = kinetic_energy
        potential_energy = 0.5 * vib_freq * physical_state[0,0]**2
        traj_data["potential_energy"][itrt,:] = potential_energy
    if data_storage_cfg.get("store_stoch_force",False):
        traj_data["stoch_force"][itrt] = stoch_force
    if data_storage_cfg.get("store_stoch_force",False):
        if adiabatic_current is not None:
            traj_data["adiabatic_current"][itrt] = adiabatic_current(physical_state[0,0])
        else:
            raise ValueError("Must provide adiabatic current")

def run_single_trajectory(
        cfg,
        phys_ic,
        determ_aux_ic,
        stoch_aux_ic,
        interp_forces_arr,
        dimensionfull_func_arr,
        el_obs_arr,
        grid_limits,
): 

    simulation_cfg = cfg["simulation"]
    data_storage_cfg = cfg["simulation"]["data_storage"]
    dt = simulation_cfg["dt"]
    n_timesteps = simulation_cfg["n_timesteps"]
    n_vib = cfg["n_vib"]
    traj_data = initialize_traj_data(data_storage_cfg,n_timesteps,n_vib)
    physical_state = phys_ic.copy()
    determ_aux_state = determ_aux_ic.copy()
    stoch_aux_state = stoch_aux_ic.copy()
    stoch_force = generate_stoch_force(stoch_aux_state)
    store_traj_data(traj_data,0,physical_state,stoch_force,cfg,
                    adiabatic_current=el_obs_arr[0])
    for itrt in range(1,n_timesteps):
        physical_state,determ_aux_state,stoch_aux_state,stoch_force = propagate_aboba_onestep(
            physical_state,
            determ_aux_state,
            stoch_aux_state,
            interp_forces_arr,
            dimensionfull_func_arr,
            dt,
            grid_limits
        )
        store_traj_data(traj_data,itrt,physical_state,stoch_force,cfg,
                        adiabatic_current=el_obs_arr[0])

    return traj_data


def run_single_trajectory_markovian(
        cfg,
        phys_ic,
        interp_forces_arr,
        dimensionfull_func_arr,
        grid_limits
): 

    simulation_cfg = cfg["simulation"]
    data_storage_cfg = cfg["simulation"]["data_storage"]
    dt = simulation_cfg["dt"]
    n_timesteps = simulation_cfg["n_timesteps"]
    n_vib = cfg["n_vib"]
    Kval = simulation_cfg["Kval"]
    traj_data = initialize_traj_data(data_storage_cfg,n_timesteps,n_vib)
    physical_state = phys_ic.copy()
    store_traj_data(traj_data,0,physical_state,0,cfg)
    for itrt in range(1,n_timesteps):
        propagate_aboba_markovian_onestep(
            physical_state,
            interp_forces_arr,
            dimensionfull_func_arr,
            dt,
            grid_limits,Kval
        )
        store_traj_data(traj_data,itrt,physical_state,0,cfg)

    return traj_data

def save_trajectory(traj_data,traj_dir,traj_id,markovian_value):

    if markovian_value:
        save_dir = traj_dir / f"traj_{traj_id}_markovian.npz"
        np.savez(save_dir, **traj_data)
    else:
        save_dir = traj_dir / f"traj_{traj_id}.npz"
        np.savez(save_dir, **traj_data)

def single_worker(
        cfg,
        traj_dir,
        traj_indices,
        interp_forces_arr,
        dimensionfull_func_arr,
        initial_conditions_arr,
        el_obs_arr,
        grid_limits,
        worker_id=None
):

    if worker_id is not None:
        np.random.seed(cfg["simulation"]["base_seed"] + worker_id)  

    markovian_value = cfg["simulation"]["markovian"]
    if markovian_value:
        for traj_id in traj_indices:
            phys_ic = initial_conditions_arr[traj_id]
            traj_data = run_single_trajectory_markovian(
                cfg,
                phys_ic,
                interp_forces_arr,
                dimensionfull_func_arr,
                grid_limits
                )
            save_trajectory(traj_data,traj_dir,traj_id,markovian_value)
            del traj_data
    else:
        for traj_id in traj_indices:
            phys_ic,determ_aux_ic,stoch_aux_ic = initial_conditions_arr[traj_id]
            traj_data = run_single_trajectory(
                cfg,
                phys_ic,
                determ_aux_ic,
                stoch_aux_ic,
                interp_forces_arr,
                dimensionfull_func_arr,
                el_obs_arr,
                grid_limits,
                )
            save_trajectory(traj_data,traj_dir,traj_id,markovian_value)
            del traj_data

def create_traj_indices(n_traj,n_cores):

    traj_ids = np.arange(n_traj)
    traj_chunks = np.array_split(traj_ids, n_cores)

    return traj_chunks

def compute_single_voltage(cfg,voltage):

    # Precompute shared data
    interp_forces_arr,el_obs_arr,grid_limits = prepare_forces.load_and_interpolate_forces(cfg,voltage)
    initial_conditions_arr = prepare_initial_conditions.generate_all_ics(cfg,interp_forces_arr)
    dimensionfull_func_arr = equations_of_motion.build_equations_of_motion(cfg)

    # Prepare trajectory information
    simulation_cfg = cfg["simulation"]
    n_traj = simulation_cfg["n_traj"]
    n_cores = simulation_cfg["n_cores"]
    traj_chunks = create_traj_indices(n_traj,n_cores)
    traj_dir = ensure_directory(Path(cfg["results_transient_root"] / 
                    cfg["system_identifier_dir"] / 
                        f"voltage_{voltage:.2f}eV"))
    
    # Spawn workers
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for worker_id,chunk in enumerate(traj_chunks):
            futures.append(
                executor.submit(
                    single_worker,
                    cfg,
                    traj_dir,
                    chunk,
                    interp_forces_arr,
                    dimensionfull_func_arr,
                    initial_conditions_arr,
                    el_obs_arr,
                    grid_limits,
                    worker_id
                )
            )
        for f in as_completed(futures):
            # Propagate exceptions if any
            f.result()