# source/langevin_propagation/trajectory_sampling.py
"""
Docstring for source.langevin_propagation.trajectory_io
"""

import numpy as np
# from pathlib import Path

class TrajectoryWriter:

    def __init__(self, results_layout, cfg):

        self.results_layout = results_layout
        self.markovian = cfg["simulation"]["markovian"]

    def save_transient(self, data, traj_id):

        fname = f"traj_transient_{traj_id}"
        if self.markovian:
            fname += "_markovian"

        np.savez(self.results_layout.transient_traj / fname, **data)
    
    def load_transient(self, traj_id):

        fname = f"traj_transient_{traj_id}"
        if self.markovian:
            fname += "_markovian"

        return np.load(self.results_layout.transient_traj / fname)

    def save_ss(self, data, traj_id):

        fname = f"ss_ic_{traj_id}"
        if self.markovian:
            fname += "_markovian"

        np.savez(self.results_layout.ss_initial_conds / fname, **data)

    def save_ss_sample(self, data, traj_id):

        fname = f"ss_sample_{traj_id}"
        if self.markovian:
            fname += "_markovian"

        np.savez(self.results_layout.ss_samples / fname, **data)

    def load_ss_ic(self, traj_id):

        fname = f"ss_ic_{traj_id}"
        if self.markovian:
            fname += "_markovian"
        fname += ".npz"
        
        with np.load(self.results_layout.ss_initial_conds / fname) as f:
            # read arrays into memory
            ss_ic = {key: f[key] for key in f.keys()}    
        
        return ss_ic
