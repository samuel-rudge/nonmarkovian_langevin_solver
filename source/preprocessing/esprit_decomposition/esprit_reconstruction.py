# source/preprocessing/lorentzian_decomposition/lorentzian_reconstruction.py
"""
Series of functions used in the high-level Lorentzian decomposition procedure

Provides:
---------
- generate_estimate_function(freq_vec,decomp_parameters)
    Given estimated poles (in Lorentzian form) and frequency vector, construct F_estimate(w), which should be 
    as close to F_target(w) as possible

- fourier_transform_function_given_poles(poles,w)
    Given estimated poles (in universal form) and frequency vector, construct F_estimate(w), which should be 
    as close to F_target(w) as possible

- generate_frequencies(cfg)
    Generate vector of logarithmically spaced frequencies upon which to do the Lorentzian fit

- poles_to_decomp_parameters(pole_list)
    Convert between forms of poles for Lorentzians + Universal

- prepare_target(cfg,input_poles)
    Generate target function given input poles

"""

import numpy as np
from source.preprocessing import pole_dataclass
from source.utils.enums import LorPoleType
from typing import Dict , List , Tuple
from numpy.typing import NDArray

def generate_estimate(
        time_arr: NDArray[np.floating],
        poles: List[pole_dataclass.Pole]
) -> NDArray[np.floating]:
    """
    Given set of poles in Lorentzian form (frequency, width, weight), construct F_estimate(w)

    Inputs
    ------
    freq_vec: NDArray[np.floating],
    
    decomp_parameters: Dict[LorPoleType, NDArray[np.floating]]

    Outputs
    -------
    estimate: NDArray[np.floating]
        Estimate function 
    """
    
    estimate = np.zeros_like(time_arr, dtype=complex)
    for k, pole in enumerate(poles):
        freq = pole.real_part_freq + 1j*pole.imag_part_freq
        weight = pole.real_part_weight + 1j*pole.imag_part_weight
        estimate += np.real(weight * np.exp(-freq * time_arr))

    return estimate

def residual(poles, time_arr, target):
    
    # Generate estimate
    estimate = generate_estimate(time_arr,poles)
    # Return real vector for least_squares
    r = residual_fraction(target,estimate)

    return r

def residual_fraction(f_target, f_estimate, eps=1e-12):
    """
    Fractional RMS error for real or complex signals.
    """
    numerator = np.linalg.norm(f_target - f_estimate)
    denominator = np.linalg.norm(f_target) + eps
    
    return numerator / denominator