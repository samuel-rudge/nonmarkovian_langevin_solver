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

def generate_lorentzian_function_estimate(
        freq_vec: NDArray[np.floating],
        decomp_parameters: Dict[LorPoleType, NDArray[np.floating]]
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
    # Broadcast poles and Fourier space frequency shapes together
    frequency = decomp_parameters[LorPoleType.FREQUENCY][None, :] # (N_terms, 1)
    width = decomp_parameters[LorPoleType.WIDTH][None, :]         # (N_terms, 1)
    weight = decomp_parameters[LorPoleType.WEIGHT][None, :]       # (N_terms, 1)
    w = freq_vec[:, None]                                         # (N_freq,1)
    
    # Construct estimate function
    lorentz_pos = weight / ((w - frequency)**2 + width**2)
    lorentz_neg = weight / ((w + frequency)**2 + width**2)
    estimate = np.sum(lorentz_pos + lorentz_neg,axis=1)

    return estimate

def generate_ft_function_given_poles(
        poles: List[pole_dataclass.Pole],
        freq_vec: NDArray[np.floating]
) -> NDArray[np.floating]:

    """
    Given set of poles in universal form (pole from Pole dataclass = (frequencies,weights)),
    construct F(w)

    Inputs
    ------
    freq_vec: NDArray[np.floating],
    
    poles: List[pole_dataclass.Pole]

    Outputs
    -------
    reconstructed_function: NDArray[np.floating]
        Reconstructed function 
    """

    freq = np.array([p.real_part_freq + 1j*p.imag_part_freq for p in poles],
                    dtype=complex)[None, :]
    weight =  np.array([p.real_part_weight + 1j*p.imag_part_weight for p in poles],
                        dtype=complex)[None, :]
    w = freq_vec[:, None]
    reconstructed_function = 2 * np.sum(np.real(weight / (freq - 1j*w)),axis=1)

    return reconstructed_function

def generate_frequencies(cfg: Dict) -> NDArray[np.floating]:

    decomposition_cfg = cfg["decomposition"]
    cutoff_factor = decomposition_cfg["cutoff_factor"]
    cutoff_frequency = decomposition_cfg["cutoff_frequency"]
    Nsupport_points = decomposition_cfg["Nsupport_points"]
    max_freq = cutoff_factor*cutoff_frequency
    freq_vec_pos = np.geomspace(max_freq*1e-10,max_freq,Nsupport_points)
    # freq_vec_pos = np.linspace(max_freq*1e-10,max_freq,Nsupport_points_aaa_initial)
    freq_vec = np.concatenate((-freq_vec_pos[::-1],[0.0],freq_vec_pos))

    return freq_vec

def poles_to_decomp_parameters(
        pole_list: List[pole_dataclass.Pole]
) -> Dict[LorPoleType, NDArray]:
    
    """
    Convert a list of Pole objects into [omega, width, weight] arrays
    for the decomposition code.
    """
    N = len(pole_list)
    omega  = np.zeros(N, dtype=float)
    width  = np.zeros(N, dtype=float)
    weight = np.zeros(N, dtype=float)

    for i, p in enumerate(pole_list):
        omega[i]  = p.imag_part_freq
        width[i]  = p.real_part_freq
        weight[i] = p.real_part_weight

    decomp_parameters: Dict[LorPoleType, np.ndarray] = {
    LorPoleType.FREQUENCY : omega,
    LorPoleType.WIDTH : width,
    LorPoleType.WEIGHT : weight
    }

    return decomp_parameters

def prepare_target(
        cfg: Dict,
        input_poles: List[pole_dataclass.Pole]
) -> Tuple[NDArray[np.floating],NDArray[np.floating],float]:

    freq_vec = generate_frequencies(cfg)
    freq_vec_pos = freq_vec[freq_vec > 0.0]
    target = generate_ft_function_given_poles(input_poles,freq_vec_pos)
    initial_norm = np.trapz(target,freq_vec_pos)
    if initial_norm == 0:
        raise ValueError("Target function has zero norm; cannot normalize.")
    target = target / initial_norm

    return freq_vec_pos,target,initial_norm
