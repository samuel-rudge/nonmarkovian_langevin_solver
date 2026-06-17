"""
Docstring for source.langevin_propagation.trajectory_propagation
"""

from source.langevin_propagation import (
    integrator_aboba,
    trajectory_sampling
)
import numpy as np
import copy

def run_transient_trajectory(
        cfg,
        voltage,
        initial_conds,
        interp_forces_arr,
        dimensionfull_func_arr,
        grid_limits,
        keep_traj,
        rng
): 

    simulation_cfg = cfg["simulation"]
    dt = simulation_cfg["dt"]
    n_timesteps = trajectory_sampling.ss_n_timesteps(cfg)
    state = copy.deepcopy(initial_conds)
    sampler_object = trajectory_sampling.TrajectorySampler(
        cfg, voltage, traj_type="transient", keep_traj=keep_traj
    )
    stoch_force = integrator_aboba.generate_stoch_force(state["stoch_aux_state"])
    observables = {
        "stoch_force": stoch_force
    }
    sampler_object.observe(state,observables,itrt=0)
    for itrt in range(1,n_timesteps):
        state,observables = integrator_aboba.propagate_aboba_onestep(
            state,
            interp_forces_arr,
            dimensionfull_func_arr,
            dt,
            grid_limits,
            rng
        )
        sampler_object.observe(state,observables,itrt=itrt)

    return sampler_object.finalize(),state

def run_ss_trajectory(
        cfg,
        voltage,
        initial_conds,
        interp_forces_arr,
        dimensionfull_func_arr,
        grid_limits,
        rng
):    

    simulation_cfg = cfg["simulation"]
    dt = simulation_cfg["dt"]
    n_timesteps = trajectory_sampling.ss_sample_timesteps(cfg)#
    state = copy.deepcopy(initial_conds)
    stoch_force = integrator_aboba.generate_stoch_force(state["stoch_aux_state"])
    observables = {
        "stoch_force": stoch_force
    }
    sampler_object = trajectory_sampling.TrajectorySampler(
        cfg, voltage, traj_type="ss", keep_traj=True
    )
    sampler_object.observe(state,observables,time=0)
    for itrt in range(1,n_timesteps):
        state,observables = integrator_aboba.propagate_aboba_onestep(
            state,
            interp_forces_arr,
            dimensionfull_func_arr,
            dt,
            grid_limits,
            rng
        )
        sampler_object.observe(state,observables,time=itrt * dt)

    return sampler_object.finalize()

def run_transient_trajectory_markovian(
        cfg,
        voltage,
        phys_ic,
        interp_forces_arr,
        dimensionfull_func_arr,
        grid_limits,
        keep_traj
): 

    simulation_cfg = cfg["simulation"]
    dt = simulation_cfg["dt"]
    n_timesteps = simulation_cfg["n_timesteps"]
    Kval = simulation_cfg["Kval"]
    state = copy.deepcopy(phys_ic)
    sampler_object = trajectory_sampling.TrajectorySampler(
        cfg, voltage, traj_type="ss", keep_traj=keep_traj
    )
    sampler_object.observe(state,itrt=0)
    for itrt in range(1,n_timesteps):
        state = integrator_aboba.propagate_aboba_markovian_onestep(
            state,
            interp_forces_arr,
            dimensionfull_func_arr,
            dt,
            grid_limits,Kval
        )
        sampler_object.observe(state,itrt=itrt)

    return sampler_object.finalize(),state

def run_ss_trajectory_markovian(
        cfg,
        voltage,
        initial_conds,
        interp_forces_arr,
        dimensionfull_func_arr,
        grid_limits
):    

    simulation_cfg = cfg["simulation"]
    dt = simulation_cfg["dt"]
    n_timesteps = trajectory_sampling.ss_sample_timesteps(cfg)
    state = copy.deepcopy(initial_conds)
    sampler_object = trajectory_sampling.TrajectorySampler(
        cfg, voltage, traj_type="ss", keep_traj=True
    )
    sampler_object.observe(state,time=0)
    for itrt in range(1,n_timesteps):
        state = integrator_aboba.propagate_aboba_markovian_onestep(
            state,
            interp_forces_arr,
            dimensionfull_func_arr,
            dt,
            grid_limits,
            Kval=10
        )
        sampler_object.observe(state,time=itrt * dt)

    return sampler_object.finalize()
