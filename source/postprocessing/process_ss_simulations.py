# source/postprocessing/process_transient_simulations.py

import numpy as np
import pandas as pd
from source.postprocessing import results_io
from source.utils.results_layout import ResultsLayout

def construct_observable_keys(cfg):

    data_storage_cfg = cfg["simulation"]["data_storage"]
    keys = {"x", "p"}
    if data_storage_cfg.get("store_energy",False):
        keys.add("kinetic_energy")
        keys.add("potential_energy")
    if data_storage_cfg.get("store_current",False):
        keys.add("adiabatic_current")
    
    return keys

def initialize_observable_stats(voltage, observables):

    data_arr = {}
    data_arr["voltage"] = voltage
    for obs in observables:
        data_arr[obs] = {
            "mean": 0.0,
            "std": 0.0,
            "stderr": 0.0,
            "M2": 0.0,
            "n": 0
        }

    return data_arr

def generate_ss_observable_stats(data_arr,ss_sample_list,results_writer,observables):
        
    for ss_sample_id in ss_sample_list:
        ss_data = results_writer.load_ss_sample(ss_sample_id)
        for key in observables:
            # data_arr[key]["mean"] += np.sum(ss_data[key],axis=0) # For 1 vib mode axis = 0
            if key == "x":
                flattened_data = ss_data["physical_state"][:,0,0].ravel()
            elif key == "p":
                flattened_data = ss_data["physical_state"][:,0,1].ravel()
            elif key == "kinetic_energy" or key == "potential_energy":
                flattened_data = ss_data[key][:,0].ravel()
            else:
                flattened_data = ss_data[key].ravel()
            for sample in flattened_data:
                stat = data_arr[key]
                stat["n"] += 1
                delta1 = sample - stat["mean"]
                stat["mean"] += delta1 / stat["n"]
                delta2 = sample - stat["mean"]
                stat["M2"] += delta1 * delta2
        ss_data.close()

    for key in observables:
        stat = data_arr[key]
        if stat["n"] > 1:
            variance = stat["M2"] / (stat["n"] - 1)
            stat["std"] = np.sqrt(variance)
            stat["stderr"] = stat["std"] / np.sqrt(stat["n"])
        else:
            stat["std"] = np.nan
        del stat["M2"]
        del stat["n"]
    
    return data_arr

def ss_stats_to_dataframe_row(data):
    
    row = {"voltage": data["voltage"]}
    for key, stats in data.items():
        if key == "voltage":
            continue
        row[f"{key}_mean"] = stats["mean"]
        row[f"{key}_std"] = stats["std"]

    return pd.DataFrame([row])


def generate_ss_av(cfg,voltage):

    results_layout = ResultsLayout(cfg,voltage)
    results_writer = results_io.ResultsWriter(results_layout,cfg)
    observables = construct_observable_keys(cfg)
    data_arr = initialize_observable_stats(voltage,observables)
    # Generate steady-state sample list
    n_traj = cfg["simulation"]["n_traj"]
    ss_sample_list = np.arange(n_traj)

    data_arr = generate_ss_observable_stats(
        data_arr,
        ss_sample_list,
        results_writer,
        observables
        )
    
    data_frame = ss_stats_to_dataframe_row(data_arr)
    results_writer.save_ss_dataframe(data_frame)
