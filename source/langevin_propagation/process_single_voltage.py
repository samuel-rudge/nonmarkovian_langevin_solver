# source/langevin_propagation/voltage_level_orchestration.py
"""
Docstring for source.langevin_propagation.process_single_voltage
"""

import numpy as np
from source.langevin_propagation import (
    prepare_forces,
    prepare_initial_conditions,
    equations_of_motion
)
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from source.utils.results_layout import ResultsLayout
from source.langevin_propagation import(
    trajectory_sampling,
    trajectory_workers
)

def compute_single_voltage_to_ss(cfg,voltage):

    # Precompute shared data
    interp_forces_arr,grid_limits = prepare_forces.load_and_interpolate_forces(cfg,voltage)
    initial_conditions_arr,traj_seeds = prepare_initial_conditions.generate_all_ics(cfg,interp_forces_arr)
    dimensionfull_func_arr = equations_of_motion.build_equations_of_motion(cfg)

    # Prepare trajectory information
    simulation_cfg = cfg["simulation"]
    traj_chunks = trajectory_workers.create_traj_indices(simulation_cfg)
    traj_to_keep,traj_to_keep_mask = trajectory_sampling.generate_traj_to_keep(simulation_cfg)
    results_layout = ResultsLayout(cfg, voltage)
    results_layout.ensure()
    replace_data = simulation_cfg["ss_processing"]["replace_data"]
    if replace_data:
        results_layout.remove_data()

    # Spawn workers
    n_cores = simulation_cfg["n_cores"]
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for worker_id,chunk in enumerate(traj_chunks):
            futures.append(
                executor.submit(
                    trajectory_workers.single_worker_to_ss,
                    cfg,
                    voltage,
                    chunk,
                    interp_forces_arr,
                    dimensionfull_func_arr,
                    initial_conditions_arr,
                    grid_limits,
                    traj_to_keep_mask,
                    traj_seeds,
                    worker_id
                )
            )
        for f in as_completed(futures):
            # Propagate exceptions if any
            f.result()

    return traj_to_keep

def compute_single_voltage_sample_ss(cfg,voltage,traj_seeds):

    # Precompute shared data
    interp_forces_arr,grid_limits = prepare_forces.load_and_interpolate_forces(cfg,voltage)
    dimensionfull_func_arr = equations_of_motion.build_equations_of_motion(cfg)

    # Prepare trajectory information
    simulation_cfg = cfg["simulation"]
    traj_chunks = trajectory_workers.create_traj_indices(simulation_cfg)
    results_layout = ResultsLayout(cfg, voltage)
    results_layout.ensure()
    # traj_dir = ensure_directory(Path(cfg["results_transient_root"] / 
    #                 cfg["system_identifier_dir"] / 
    #                     f"voltage_{voltage:.2f}eV"))
    
    # Spawn workers
    n_cores = simulation_cfg["n_cores"]
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for worker_id,chunk in enumerate(traj_chunks):
            futures.append(
                executor.submit(
                    trajectory_workers.single_worker_sample_ss,
                    cfg,
                    voltage,
                    chunk,
                    interp_forces_arr,
                    dimensionfull_func_arr,
                    grid_limits,
                    traj_seeds,
                    worker_id
                )
            )
        for f in as_completed(futures):
            # Propagate exceptions if any
            f.result()