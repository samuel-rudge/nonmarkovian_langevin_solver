# source/langevin_propagation/trajectory_sampling.py
"""
Docstring for source.langevin_propagation.trajectory_io
"""

import numpy as np
from pathlib import Path
from source.utils.file_walker import iter_voltage_dirs
from source.utils.results_layout import ResultsLayout
import pandas as pd 

def combine_ss_results(cfg):

    results_layout = ResultsLayout(cfg,0.0)
    total_ss_results_dir = results_layout.base
    min_voltage = cfg["voltage"]["min"]
    max_voltage = cfg["voltage"]["max"]
    step_voltage = cfg["voltage"]["step"]
    n_steps = int(round((max_voltage - min_voltage) / step_voltage))
    voltages = [
        round(min_voltage + i * step_voltage, 10)
        for i in range(n_steps + 1)
    ]
    dfs = []
    for voltage in voltages:
        results_layout_voltage = ResultsLayout(cfg,voltage)
        df = pd.read_csv(results_layout_voltage.ss_av)
        df["voltage"] = voltage
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(total_ss_results_dir / f"ss_av.csv", index=False)


class ResultsWriter:

    def __init__(self, results_layout, cfg):

        self.results_layout = results_layout
        self.markovian = cfg["simulation"]["markovian"]

    def load_transient(self, traj_id):

        fname = f"traj_transient_{traj_id}"
        if self.markovian:
            fname += "_markovian"
        fname += ".npz"

        return np.load(self.results_layout.transient_traj / fname)

    def load_ss_sample(self, ss_sample_id):

        fname = f"ss_sample_{ss_sample_id}"
        if self.markovian:
            fname += "_markovian"
        fname += ".npz"

        return np.load(self.results_layout.ss_samples / fname)

    def save_ensemble_av_data(self,data):

        fname = f"ensemble_av"
        if self.markovian:
            fname += "_markovian"
        fname += f".npz"
        
        np.savez(self.results_layout.transient_ensemble_av / fname, **data)
    
    def save_ss_dataframe(self,data_frame):
        
        fname = f"ss_av"
        if self.markovian:
            fname += "_markovian"
        fname += ".csv"
        # csv_path = self.results_layout.ss_av / fname
        data_frame.to_csv(self.results_layout.ss_av / fname, index=False)
        # if csv_path.exists():
        #     df_existing = pd.read_csv(csv_path)
        #     df = pd.concat([df_existing, df_row], ignore_index=True)
        # else:
        #     df = df_row

        # df.to_csv(csv_path, index=False)
