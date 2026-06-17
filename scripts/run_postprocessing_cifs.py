# scripts/run_preprocessing.py
"""
Script to run the full preprocessing pipeline, which
    - scrapes the forces from all parameters/voltages in the results directory
      specified in config, in an exploratory way
    - loads the time-dependent friction and correlation function for each x-value 
    - uses ESPRIT to decompose the friction and correlation function at each
      x-value in terms of a set number of frequencies and weights
    - collects, orders, and saves these into arrays
    - plots the frequencies and weights as functions of x

USAGE:
------
python3 -m scripts.run_preprocessing
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
