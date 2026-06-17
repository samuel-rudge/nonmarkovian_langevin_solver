# source/utils/results_layout.py

from pathlib import Path
from source.utils.io_utils import ensure_directory,clear_directory
from decimal import Decimal

TWO_DP = Decimal("0.00")
THREE_DP = Decimal("0.000")
TENTH = Decimal("0.05")

def format_voltage(v: Decimal) -> str:
    if (v % TENTH) == 0:
        return f"{v.quantize(TWO_DP)}"
    else:
        return f"{v.quantize(THREE_DP)}"

class ResultsLayout:

    def __init__(self, cfg, voltage):
        if not isinstance(voltage, Decimal):
            voltage = Decimal(str(voltage))


        self.base = Path(cfg["results_root"] / cfg["system_identifier_dir"])
        self.voltage_dir = self.base / f"voltage_{format_voltage(voltage)}eV"

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