# scripts/ss_postprocessing_pipeline.py
"""
Steady-state postprocessing pipeline for Langevin dynamics simulations.

This script performs postprocessing of trajectory data generated during
steady-state propagation runs. It operates on results that already exist
on disk and does not perform any further propagation.

For a given bias voltage, the postprocessing pipeline:
    - loads the simulation configuration
    - reads transient trajectory samples retained during propagation
    - computes ensemble-averaged transient observables (if available)
    - generates diagnostic plots to assess steady-state convergence
    - computes steady-state ensemble averages of observables
    - writes a single steady-state summary row to disk

Transient postprocessing is optional and failure-tolerant: if transient
trajectory data or plots cannot be generated, the script will continue
and still perform steady-state averaging.

In addition, the script can collect steady-state results across all
voltages into a single combined dataset.

USAGE
-----
Postprocess steady-state results for a single voltage:
    python3 -m scripts.ss_postprocessing_pipeline --voltage "<value>eV"

Collect steady-state results across all voltages:
    python3 -m scripts.ss_postprocessing_pipeline --collect

BEHAVIOR
--------
Single-voltage postprocessing performs:
    - transient ensemble-averaged plotting (best-effort)
    - steady-state ensemble averaging (mandatory)

Collected steady-state data are written to:
    results/.../ss_av.csv          (per voltage)
    results/.../ss_av_combined.csv (after --collect)

NOTES
-----
- This script assumes that steady-state propagation has already been
  completed for the requested voltage(s).
- Transient plots are intended as diagnostics to verify convergence
  (e.g. kinetic and potential energy saturation).
- The --collect option should be run only after all voltage jobs have
  completed successfully.

See also:
    - docs/postprocess.md
    - docs/propagate.md
    - docs/hpc.md
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
        process_transient_simulations.calculate_ensemble_averages(cfg,
                                                              voltage,
                                                              traj_list=None,
                                                              traj_type="ss"
                                                              )
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
