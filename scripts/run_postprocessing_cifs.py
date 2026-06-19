# scripts/run_postprocessing_cifs.py
"""
Electronic forces postprocessing pipeline.

This script postprocesses *precomputed electronic force datasets* for one or
more bias voltages. It operates exclusively on data generated during the
preprocessing stage and stored in the `results/` directory.

For each voltage, the script:
    - loads x-dependent adiabatic forces, Markovian friction, and
      non-Markovian force-force correlation functions
    - decomposes time-dependent friction and correlation functions at each
      nuclear coordinate using ESPRIT into a discrete set of frequencies
      and weights
    - collects and orders the extracted poles consistently across the
      nuclear coordinate grid
    - generates diagnostic plots of:
        * Markovian forces and friction as functions of x
        * non-Markovian mode frequencies vs x
        * non-Markovian mode weights vs x
    - saves all plots as PDF files in:
        results / <identifier> / voltage_<value>eV / el_forces /

The decomposition is performed independently at each nuclear configuration,
and no dynamical propagation is involved in this step.

This script is intended for:
    - validating electronic-force datasets
    - inspecting x-dependent structure in friction and noise
    - diagnosing nonequilibrium effects as a function of bias voltage

USAGE
-----
Postprocess electronic forces for all voltages:
    python3 -m scripts.run_postprocessing_cifs

Postprocess electronic forces for a single voltage:
    python3 -m scripts.run_postprocessing_cifs --voltage "<value>eV"

NOTES
-----
- Preprocessing must be completed before running this script.
- This script performs no data modification; it only reads existing results
  and generates plots.
- All output is written to voltage-resolved subdirectories under `results/`.

See also:
    - docs/electronic_force_dataset_spec.md
    - docs/preprocess.md
    - docs/postprocess.md
"""

from source.utils.config import load_config 
from source.postprocessing.plotting_pole_decomp_over_grid import (
    generate_all_plots,
    generate_all_plots_single_voltage
    )

from source.utils.file_walker import generate_voltage_dir
from source.utils.results_layout import ResultsLayout
from pathlib import Path
import argparse
from decimal import Decimal

def postprocessing_el_forces_script():

  cfg = load_config()
  generate_all_plots(cfg)

def postprocessing_el_forces_single_voltage_script(voltage):

  cfg = load_config()
  results_layout = ResultsLayout(cfg,voltage)
  generate_all_plots_single_voltage(cfg,results_layout.voltage_dir)

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
        postprocessing_el_forces_script()
    else:
        postprocessing_el_forces_single_voltage_script(args.voltage)

if __name__ == "__main__":

  main()
