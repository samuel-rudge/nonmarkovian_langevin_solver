# source/config/types.py

from typing import (
    Dict, 
    List, 
    Tuple,
    TypedDict
)
from numpy.typing import NDArray # type: ignore
import numpy as np

class CIFSResults(TypedDict):

    x: NDArray[np.floating]
    freq_real: NDArray[np.floating]
    freq_imag: NDArray[np.floating]
    weights_real: NDArray[np.floating]
    weights_imag: NDArray[np.floating]
    weights_mag: NDArray[np.floating]
    weights_phase: NDArray[np.floating]
