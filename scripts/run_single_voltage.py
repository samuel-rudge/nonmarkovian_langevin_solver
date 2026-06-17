# scripts/run_single_voltage.py

from source.langevin_propagation.transient_pipeline import compute_single_voltage
from source.utils.config import load_config

def main():
    
    cfg = load_config()
    voltage = cfg["voltage"]["min"]
    compute_single_voltage(cfg,voltage)

if __name__ == "__main__":

    main()