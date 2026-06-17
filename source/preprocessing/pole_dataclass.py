# source/preprocessing/pole_dataclass.py
"""
Defines the dataclass that we use to store the weights and frequencies (poles) of our 
decompositions

Provides:
---------
- pole_dataclass.Pole
    Given a function evaluated on a uniformly spaced interval, calculates and 
    returns the weights and frequencies from an ESPRIT decomposition of this function.

Usage:
------
from source.preprocessing.pole_dataclass import Pole

poles = Pole(f,w)
------
"""

import numpy as np
from dataclasses import dataclass
from dataclasses import dataclass
import numpy as np


@dataclass
class Pole:
    """One pole of the decomposition at a single x-coordinate"""
    frequency: complex
    weight: complex

    # -------------------------
    # Frequency accessors
    # -------------------------
    @property
    def real_part_freq(self) -> float:
        return float(np.real(self.frequency))

    @real_part_freq.setter
    def real_part_freq(self, value: float):
        self.frequency = complex(value, self.imag_part_freq)

    @property
    def imag_part_freq(self) -> float:
        return float(np.imag(self.frequency))

    @imag_part_freq.setter
    def imag_part_freq(self, value: float):
        self.frequency = complex(self.real_part_freq, value)

    # -------------------------
    # Weight accessors
    # -------------------------
    @property
    def real_part_weight(self) -> float:
        return float(np.real(self.weight))

    @real_part_weight.setter
    def real_part_weight(self, value: float):
        self.weight = complex(value, self.imag_part_weight)

    @property
    def imag_part_weight(self) -> float:
        return float(np.imag(self.weight))

    @imag_part_weight.setter
    def imag_part_weight(self, value: float):
        self.weight = complex(self.real_part_weight, value)

    # -------------------------
    # Utilities
    # -------------------------
    def __repr__(self):
        return (
            f"Pole(f={self.frequency:.3g}, "
            f"w={self.weight:.3g}, "
            f"Re(f)={self.real_part_freq:.3g})"
        )

    def distance(self, other: "Pole", alpha: float = 1.0) -> float:
        return (
            abs(self.real_part_freq - other.real_part_freq)
            + alpha * abs(self.weight - other.weight)
        )

def create_pole_arr_single_x(poles):

    pole_arr = []
    for itrp,pole in enumerate(poles):
        pole_arr.append(pole.real_part_freq + 1j*pole.imag_part_freq)
    
    return np.array(pole_arr)

def create_poles_from_arrays(frequencies,weights):

    poles = [Pole(f,w) for f,w in zip(frequencies,weights)]
    sorted_poles = sorted(poles, key=lambda p: p.real_part_freq)

    return sorted_poles