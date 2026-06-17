# scripts/run_single_voltage.py
"""
Docstring for ss_pipeline
"""

from source.postprocessing import (
   process_transient_simulations,
   process_ss_simulations,
   plot_transient_ensemble_averages,
   results_io
)
from source.utils.config import load_config
import numpy as np
import argparse
import traceback
from decimal import Decimal

def collect_all_voltages():

    cfg = load_config()
    results_io.combine_ss_results(cfg)

def postprocessing_ss_all_voltages():

    cfg = load_config()

    # process_transient_simulations.calculate_ensemble_averages(cfg,
    #                                                           voltage,
    #                                                           traj_list=None,
    #                                                           traj_type="ss"
    #                                                           )
    # plot_transient_ensemble_averages.generate_plots_single_voltage(cfg,voltage)
    # process_ss_simulations.generate_ss_av(cfg,voltage)

def postprocessing_ss_single_voltage(voltage):

    cfg = load_config()

    try:
        # process_transient_simulations.calculate_ensemble_averages(cfg,
        #                                                       voltage,
        #                                                       traj_list=None,
        #                                                       traj_type="ss"
        #                                                       )
        plot_transient_ensemble_averages.generate_plots_single_voltage(cfg,voltage)
    except Exception as e:
        print(
                f"[WARNING] Transient postprocessing failed at voltage={voltage}. Continuing with steady-state analysis."
                )
        print(f"Reason: {e}")
        traceback.print_exc()
    # ------ steady-state postprocessing (mandatory) -------
    process_ss_simulations.generate_ss_av(cfg,voltage)

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
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Combine steady-state results across all voltages"
    )

    args = parser.parse_args()

    if args.collect:
        collect_all_voltages()
    elif args.voltage is None:
        postprocessing_ss_all_voltages()
    else:
        postprocessing_ss_single_voltage(args.voltage)

if __name__ == "__main__":

    main()
