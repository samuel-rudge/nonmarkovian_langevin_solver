# scripts/run_preprocessing.py
"""
Electronic-force preprocessing pipeline.

This script constructs the electronic force datasets required for Langevin
propagation. It operates on raw electronic data stored on disk and produces
voltage-resolved, x-dependent force representations used during dynamical
simulations.

For each bias voltage, the preprocessing stage:
    - loads raw electronic observables on a nuclear coordinate grid
    - constructs x-dependent adiabatic forces, friction kernels, and
      force-force correlation functions
    - treats Markovian and non-Markovian contributions on equal footing
    - decomposes time-dependent friction and correlation functions at each
      nuclear coordinate using ESPRIT into a finite set of modes
      (frequencies and weights)
    - orders and stores the extracted modes consistently across the grid
    - writes all processed datasets to the results directory
    - generates diagnostic plots of forces, frequencies, and weights

The resulting datasets are subsequently interpolated and queried during
trajectory propagation.

This script must be run before any propagation or postprocessing steps.

USAGE
-----
Preprocess electronic-force datasets for all voltages:
    python3 -m scripts.run_preprocessing

Preprocess electronic-force datasets for a single voltage:
    python3 -m scripts.run_preprocessing --voltage "<value>eV"

NOTES
-----
- Preprocessing is independent of trajectory propagation.
- The computational cost scales with the number of grid points and
  decomposition parameters, but is typically much cheaper than propagation.
- Output is written to voltage-resolved subdirectories under `results/`.

See also:
    - docs/preprocess.md
    - docs/electronic_force_dataset_spec.md
    - docs/propagate.md
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
