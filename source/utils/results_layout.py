# source/utils/results_layout.py

from pathlib import Path
from source.utils.io_utils import ensure_directory,clear_directory

class ResultsLayout:

    def __init__(self, cfg, voltage):

        self.base = Path(cfg["results_root"] / cfg["system_identifier_dir"])
        self.voltage_dir = self.base / f"voltage_{voltage:.2f}eV"

        self.transient = self.voltage_dir / "transient"
        self.transient_traj = self.transient / "trajectories"
        self.transient_ensemble_av = self.transient / "ensemble_av"
        self.transient_plots = self.transient / "plots"

        self.ss = self.voltage_dir / "ss"
        self.ss_samples = self.ss / "ss_samples"
        self.ss_av = self.ss / "ss_av"
        self.ss_initial_conds = self.ss / "initial_conds"

    def ensure(self):
        for p in [
            self.transient_traj,
            self.transient_plots,
            self.transient_ensemble_av,
            self.ss_samples,
            self.ss_initial_conds,
            self.ss_av
        ]:
            ensure_directory(p)

    def remove_data(self):

        for p in [
            self.transient_traj,
            self.transient_plots,
            self.transient_ensemble_av,
            self.ss_samples,
            self.ss_initial_conds,
            self.ss_av
        ]:
            clear_directory(p)