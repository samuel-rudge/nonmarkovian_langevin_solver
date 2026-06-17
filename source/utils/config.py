# source/utils/config.py
"""
config.py

Loads the project configuration from YAML

Provides:
- load_config(): returns a dictionary containing paths and parameters
- Automatically builds absolute paths for source, data, and results directories
"""

import yaml
from pathlib import Path

def load_config(path=None):
    if path is None:
        # Compute default path relative to the repo root
        repo_root = Path(__file__).resolve().parent.parent.parent
        path = repo_root / "config" / "settings.yaml"
    
    with open(path,"r") as file:
        cfg = yaml.safe_load(file)
    
    # Build absolute paths
    mollead_coupling_temperature_stem = f"gamma_{cfg['molecule_lead_coupling']}meV_temp_{cfg['temperature']}K"
    Delta_stem = f"delta_{cfg['Delta']}eV"
    tier_stem = f"tier_{cfg['electronic_tier']}"
    cfg["source_root"] =  Path(cfg["source_root"])
    cfg["system_identifier_dir"] = Path(cfg["data_identifier"]) / mollead_coupling_temperature_stem / Delta_stem / tier_stem
    cfg["data_root"] = Path(cfg["data_root"])
    cfg["raw_data_root"] =  cfg["data_root"] / cfg["system_identifier_dir"]
    cfg["results_root"] =  Path(cfg["results_root"])
    cfg["el_forces_root"] =  Path(cfg["results_subdirs"]["el_forces"])
    cfg["results_ss_root"] =  Path(cfg["results_root"]) / Path(cfg["results_subdirs"]["steady_state"])
    cfg["results_transient_root"] =  Path(cfg["results_root"]) / Path(cfg["results_subdirs"]["transient"])
    
    return cfg