# scripts/run_single_voltage.py
"""
Docstring for ss_pipeline
"""
from source.langevin_propagation.process_single_voltage import (
    compute_single_voltage_to_ss,
    compute_single_voltage_sample_ss
)
from source.postprocessing import (
   process_transient_simulations,
   process_ss_simulations,
   plot_transient_ensemble_averages
)
from source.utils.config import load_config
import numpy as np
import argparse
from decimal import Decimal

def run_ss_all_voltages():

    cfg = load_config()
    min_voltage = cfg["voltage"]["min"]
    max_voltage = cfg["voltage"]["max"]
    dvoltage = cfg["voltage"]["step"]
    for voltage in np.arange(min_voltage,max_voltage,dvoltage):
        run_ss_single_voltage(voltage)

def run_ss_single_voltage(voltage):

    cfg = load_config()

    traj_retained = compute_single_voltage_to_ss(cfg,voltage)
    process_transient_simulations.calculate_ensemble_averages(cfg,
                                                              voltage,
                                                              traj_list=traj_retained,
                                                              traj_type="ss"
                                                              )
    plot_transient_ensemble_averages.generate_plots_single_voltage(cfg,voltage)
    # compute_single_voltage_sample_ss(cfg,voltage)
    # process_ss_simulations.generate_ss_av(cfg,voltage)

def main():
    
    parser = argparse.ArgumentParser(
        description="Run preprocessing for all voltages or a single voltage."
    )
    parser.add_argument(
        "--voltage",
        type=Decimal,
        default=None,
        help="If provided, preprocess only this voltage"
    )

    args = parser.parse_args()

    if args.voltage is None:
        run_ss_all_voltages()
    else:
        run_ss_single_voltage(args.voltage)

if __name__ == "__main__":

    main()