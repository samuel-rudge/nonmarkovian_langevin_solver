# source/langevin_propagation/trajectory_sampling.py
"""
Docstring for source.langevin_propagation.trajectory_sampling
"""

import numpy as np
from typing import Optional , Dict
from source.langevin_propagation.prepare_forces import load_and_interpolate_el_obs

def generate_traj_to_keep(simulation_cfg):

    n_traj = simulation_cfg["n_traj"]
    percentage_to_keep = simulation_cfg["ss_processing"]["percentage_traj_to_keep"]
    n_keep = int(percentage_to_keep * n_traj)
    traj_to_keep = np.random.choice(np.arange(n_traj),n_keep,replace=False)
    traj_to_keep_mask = np.zeros(n_traj,dtype=bool)
    traj_to_keep_mask[traj_to_keep] = True

    return traj_to_keep,traj_to_keep_mask

def ss_n_timesteps(cfg):

    ss_time = cfg["simulation"]["ss_processing"]["ss_time"]
    dt = cfg["simulation"]["dt"]
    n_ss_timesteps = int(ss_time / dt)

    return n_ss_timesteps

def ss_sample_timesteps(cfg):

    dt = cfg["simulation"]["dt"]
    dt_ind = cfg["simulation"]["ss_processing"]["dt_ind"]
    n_sample = cfg["simulation"]["ss_processing"]["n_sample"]
    total_sample_time = (n_sample - 1) * dt_ind
    n_timesteps = int(total_sample_time / dt)

    return n_timesteps

class TrajectorySampler:

    def __init__(self, cfg, voltage, traj_type, keep_traj=True):

        self.cfg = cfg
        self.traj_type = traj_type
        self.el_obs_arr = load_and_interpolate_el_obs(cfg,voltage)
        self.keep_traj = keep_traj
        self.vib_freq = cfg["vib_freq"]

        self.simulation_cfg = cfg["simulation"]
        self.markovian_value = self.simulation_cfg["markovian"]
        self.data_storage_cfg = self.simulation_cfg["data_storage"]
        self.dt = self.simulation_cfg["dt"]
        if self.traj_type == "ss":
            self.dt_ind = self.simulation_cfg["ss_processing"]["dt_ind"]
            self.prev_sample_time = 0
            self.storage_ind = 0

        self._initialize_storage()
        if self.keep_traj:
            if self.traj_type == "transient":
                self.storage_ind = 0

    def _initialize_storage(self):
        # cfg,state,el_obs_arr,keep_traj,traj_type):

        if self.traj_type == "transient":
            n_timesteps = ss_n_timesteps(self.cfg)
        elif self.traj_type == "ss":
            n_timesteps = self.simulation_cfg["ss_processing"]["n_sample"]
        else:
            raise ValueError("Trajectory type must be be transient or ss")
        
        if self.keep_traj:
            data_storage_cfg = self.simulation_cfg["data_storage"]
            n_vib = self.cfg["n_vib"]
            self.data = {
                "physical_state": np.zeros((n_timesteps,n_vib,2), dtype=float)
            }
            if data_storage_cfg.get("store_energy",False):
                self.data["kinetic_energy"] = np.zeros((n_timesteps,n_vib), dtype=float)
                self.data["potential_energy"] = np.zeros((n_timesteps,n_vib), dtype=float)
            if data_storage_cfg.get("store_current",False):
                self.data["adiabatic_current"] = np.zeros((n_timesteps), dtype=float)
            if not self.markovian_value:
                if data_storage_cfg.get("store_stoch_force",False):
                    self.data["stoch_force"] = np.zeros((n_timesteps), dtype=float)
            # if markovian_value:
            #     stoch_force = None
            # else:
            #     stoch_force = generate_stoch_force(state["stoch_aux_state"])
            # # Store initial conditions
            # store_traj_data(cfg,traj_data,0,state["physical_state"],markovian_value,
            #                 stoch_force=stoch_force,adiabatic_current=el_obs_arr[0]
            #                 )
        else:
            self.data = None

    def observe(self, state, 
                observables: Optional[Dict] = None, 
                itrt: Optional[int] = None,
                time: Optional[float]=None) -> None:

        if not self.keep_traj:
            return 
        
        if self.traj_type == "ss":
            assert time is not None, "time must be provided for ss sampling"
            if (time - self.prev_sample_time) < self.dt_ind:
                return
            self.storage_ind += 1
            itrt = self.storage_ind
            self.prev_sample_time = time

        if not self.markovian_value:
            self._store(itrt, state, observables)
        else:
            self._store(itrt, state)
        
    def _store(self, itrt, state, observables=None):
        # extract physical_state, stochastic_force, el_obs etc
        assert self.data is not None, "must construct data storage container first"
        assert itrt is not None

        self.data["physical_state"][itrt,:] = state["physical_state"]
        if self.data_storage_cfg.get("store_energy",False):
            kinetic_energy = 0.5 * self.vib_freq * state["physical_state"][0,1]**2
            self.data["kinetic_energy"][itrt,:] = kinetic_energy
            potential_energy = 0.5 * self.vib_freq * state["physical_state"][0,0]**2
            self.data["potential_energy"][itrt,:] = potential_energy
        if not self.markovian_value:
            if self.data_storage_cfg.get("store_stoch_force",False):
                if observables is not None:
                    self.data["stoch_force"][itrt] = observables["stoch_force"]
                else:
                    raise ValueError("Must provide stochastic force")
        if self.data_storage_cfg.get("store_current",False):
            self.data["adiabatic_current"][itrt] = self.el_obs_arr[0](state["physical_state"][0,0])

    def finalize(self):

        return self.data