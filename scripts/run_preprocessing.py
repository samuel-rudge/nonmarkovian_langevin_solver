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
from source.preprocessing.preprocessing_pipeline import (
    run_preprocessing_pipeline,
    process_single_voltage
    )
from source.utils.file_walker import generate_voltage_dir
from pathlib import Path
import argparse
from decimal import Decimal

def preprocessing_script():

  cfg = load_config()
  run_preprocessing_pipeline(cfg)

def preprocessing_single_voltage_script(voltage):

  cfg = load_config()
  data_root = Path(cfg["raw_data_root"])
  voltage_dir = generate_voltage_dir(data_root,voltage)
  process_single_voltage(voltage_dir,cfg)

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
        preprocessing_script()
    else:
        preprocessing_single_voltage_script(args.voltage)

if __name__ == "__main__":

  main()
