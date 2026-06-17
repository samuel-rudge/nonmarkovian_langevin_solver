"""
Docstring
"""
from source.langevin_propagation import (
    trajectory_propagation,
    trajectory_propagation,
    trajectory_io
)
from source.utils.results_layout import ResultsLayout
import numpy as np

def single_worker_to_ss(
        cfg,
        voltage,
        traj_indices,
        interp_forces_arr,
        dimensionfull_func_arr,
        initial_conditions_arr,
        grid_limits,
        traj_to_keep_mask,
        traj_seeds,
        worker_id=None
):

    results_layout = ResultsLayout(cfg,voltage)
    trajectory_writer = trajectory_io.TrajectoryWriter(results_layout,cfg)
    markovian_value = cfg["simulation"]["markovian"]
    for traj_id in traj_indices:
        initial_conds = initial_conditions_arr[traj_id]
        keep_traj = traj_to_keep_mask[traj_id]
        rng = np.random.default_rng(traj_seeds[traj_id])
        if markovian_value:
            traj_data,steady_state = trajectory_propagation.run_transient_trajectory_markovian(
                    cfg,
                    voltage,
                    initial_conds,
                    interp_forces_arr,
                    dimensionfull_func_arr,
                    grid_limits,
                    keep_traj
                    )
        else:
            traj_data,steady_state = trajectory_propagation.run_transient_trajectory(
                cfg,
                voltage,
                initial_conds,
                interp_forces_arr,
                dimensionfull_func_arr,
                grid_limits,
                keep_traj,
                rng
            )
        if keep_traj:
            trajectory_writer.save_transient(traj_data,traj_id)
        del traj_data
    
        trajectory_writer.save_ss(steady_state,traj_id)

def single_worker_sample_ss(
        cfg,
        voltage,
        traj_indices,
        interp_forces_arr,
        dimensionfull_func_arr,
        grid_limits,
        traj_seeds,
        worker_id=None
):
    
    # if worker_id is not None:
    #     np.random.seed(cfg["simulation"]["base_seed"] + worker_id)  

    # Construct results layout
    results_layout = ResultsLayout(cfg,voltage)
    trajectory_writer = trajectory_io.TrajectoryWriter(results_layout,cfg)
    markovian_value = cfg["simulation"]["markovian"]
    for traj_id in traj_indices:
        rng = np.random.default_rng(traj_seeds[traj_id])
        initial_conds = trajectory_writer.load_ss_ic(traj_id)
        if markovian_value:
            ss_data = trajectory_propagation.run_ss_trajectory_markovian(
                cfg,
                voltage,
                initial_conds,
                interp_forces_arr,
                dimensionfull_func_arr,
                grid_limits
                )
            trajectory_writer.save_ss_sample(ss_data,traj_id)
            del ss_data
        else:
            ss_data = trajectory_propagation.run_ss_trajectory(
                    cfg,
                    voltage,
                    initial_conds,
                    interp_forces_arr,
                    dimensionfull_func_arr,
                    grid_limits,
                    rng
                    )
            trajectory_writer.save_ss_sample(ss_data,traj_id)

def create_traj_indices(simulation_cfg):

    n_traj = simulation_cfg["n_traj"]
    traj_ids = np.arange(n_traj)
    n_cores = simulation_cfg["n_cores"]
    traj_chunks = np.array_split(traj_ids, n_cores)

    return traj_chunks
