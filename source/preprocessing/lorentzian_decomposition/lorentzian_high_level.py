# source/preprocessing/lorentzian_decomposition/lorentzian_high_level.py
"""
Given a function in frequency space F(w) (typically defined via a sum over exponentials in 
time space), these functions fit F(w) to a sum of Lorentzian functions. Each Lorentzian 
has a positive real weight, and the decomposition can be performed either exploratively 
(without an initial guess) or using a provided initial guess. 

Provides:
---------
- lorentzian_decomposition(cfg,input_poles):
    Given a set of weights and frequencies (input_poles generated from ESPRIT or AAA)
    that define a function F(w) in frequency space, generate F(w) and perform an 
    explorative Lorentzian pole decomposition on it, i.e. fitting F(w) to a sum of 
    Lorentzians with real weights and a fixed number of terms. Outputs the corresponding 
    weights, widths, and frequencies in the same format as the input poles. The explorative 
    part refers to the fact that we estimate in an iterative fashion where the Lorentzians
    should be/how wide they should be, and then fit them via the optimization procedure.
- lorentzian_decomposition_w_input(cfg,input_poles,initial_guess):
    Does the same thing, but simply optimizes given an initial guess. 
- _generate_residual and _generate_error
    Internal helper functions for computing residuals and errors.
    
Usage:
------
from source.preprocessing.lorentzian_decomposition import lorentzian_high_level

# Explorative decomposition
poles = lorentzian_high_level(cfg,input_poles)

# Decomposition with an initial guess
poles = lorentzian_high_level(cfg,input_poles,initial_guess)
"""

import numpy as np
from source.preprocessing import pole_dataclass
from source.preprocessing.lorentzian_decomposition import (
    lorentzian_reconstruction,
    lorentzian_initialization,
    lorentzian_optimization
)
from source.utils.enums import LorPoleType
from typing import (
    Dict,
    List,
    Tuple,
    Optional
)
from numpy.typing import NDArray

def _generate_residual(
        target: NDArray[np.floating],
        estimate: NDArray[np.floating]
) -> NDArray[np.floating]:

    return target - estimate

def _generate_error(
        target: NDArray[np.floating],
        estimate: NDArray[np.floating],
        freq_vec: NDArray[np.floating]
) -> Tuple[float,float]:

    eps = 1e-12 * np.max(np.abs(target))
    rel_err = np.abs(target - estimate) / np.maximum(np.abs(target), eps)
    err_log = np.mean(rel_err)
    err_lin = np.trapz(np.abs(target - estimate), freq_vec)

    return err_log,err_lin

def lorentzian_decomposition(
    cfg: Dict,
    input_poles: List[pole_dataclass.Pole],
    time_vec: NDArray[np.floating]
) -> List[pole_dataclass.Pole]:
    """
    Perform an explorative Lorentzian decomposition of a frequency-space function.

    Starting from a set of input poles (e.g., from ESPRIT or AAA), this function generates 
    the target F(w), iteratively estimates new Lorentzian positions and widths, and then 
    optimizes all parameters to reconstruct F(w) as a sum of Lorentzians with positive weights.
    
    Input
    ----------
    cfg: Dict
        Dictionary of universal parameters
    input_poles: List[pole_dataclass.Pole]
        Poles used to construct target function F(w) to decompose
    
    Output
    ------
    poles: List[pole_dataclass.Pole]
        Optimized poles with real, positive weights to reconstruct F(w) via a sum over 
        Lorentzians
    """
    # Generate normalized target function F(w) and vector of frequencies w
    freq_vec_pos,target,initial_norm = lorentzian_reconstruction.prepare_target(cfg,input_poles)
    estimate = np.zeros_like(target)
    residual = _generate_residual(target, estimate)
    # Estimate weight, width, and frequency of first Lorentzian
    decomp_parameters = lorentzian_initialization.estimate_new_pole(
        target,
        residual,
        freq_vec_pos
    )
    # Fit via constrained least-squares to target function with constraint that the 
    # weight is real and positive
    decomp_parameters = lorentzian_optimization.generate_optimized_parameters(
        freq_vec_pos,
        target,
        decomp_parameters,
        time_vec
    )
    # Reconstruct estimate of F(w) using this optimized pole
    estimate = lorentzian_reconstruction.generate_lorentzian_function_estimate(
        freq_vec_pos,
        decomp_parameters
    )
    # Calculate residual between target and estimate
    # residual = _generate_residual(target,estimate) 
    decomposition_cfg = cfg["decomposition"]
    max_terms = decomposition_cfg["max_lorentzian_terms"]
    # tol_lorentzian = float(decomposition_cfg["tol_lorentzian"])
    # Iterate this procedure until n_terms number of Lorentzians has been reached
    for itr_term in np.arange(max_terms-1):
        # Estimate weight, width, and frequency of next Lorentzian
        decomp_parameters_new_guess = lorentzian_initialization.estimate_new_pole(
            target,
            residual,
            freq_vec_pos
            )
        if decomp_parameters_new_guess[LorPoleType.FREQUENCY] is None:
            break
        # Append this pole to set
        for lpt in LorPoleType:
            decomp_parameters[lpt] = np.append(
                decomp_parameters[lpt],
                decomp_parameters_new_guess[lpt]
                )
        # Optimize new set via consrained least-squares fit
        decomp_parameters = lorentzian_optimization.generate_optimized_parameters(
            freq_vec_pos,
            target,
            decomp_parameters,
            time_vec
        )
        # Calculate estimate and residual
        # estimate = lorentzian_reconstruction.generate_estimate_function(freq_vec_pos,decomp_parameters)
        # residual = _generate_residual(target,estimate) 
        # err_log,err_lin = _generate_error(target,estimate,freq_vec_pos)
        # print(err_log)
        # if err_log < tol_lorentzian:
        #     break

    # Convert optimized set of poles into universal dataclass form
    frequencies = (decomp_parameters[LorPoleType.WIDTH] + 
                   1j*decomp_parameters[LorPoleType.FREQUENCY])
    weights = decomp_parameters[LorPoleType.WEIGHT]*initial_norm
    poles = pole_dataclass.create_poles_from_arrays(frequencies,weights)

    return poles

def lorentzian_decomposition_w_input(
    cfg: Dict,
    input_poles: List[pole_dataclass.Pole],
    initial_guess: List[pole_dataclass.Pole],
    time_vec: NDArray[np.floating],
    center_poles: List[pole_dataclass.Pole]
) -> List[pole_dataclass.Pole]:
    
    """
    Perform a Lorentzian decomposition starting from a provided initial guess.

    Given input poles and a dictionary of initial guesses for frequencies, widths, and 
    weights, this function optimizes the Lorentzian parameters to fit F(w) without 
    iteratively estimating new poles.
    
    Input
    ----------
    cfg: Dict
        Dictionary of universal parameters
    input_poles: List[pole_dataclass.Pole]
        Poles used to construct target function F(w) to decompose
    initial_guess: List[pole_dataclass.Pole]
        Poles used for the initial guess
    
    Output
    ------
    poles: List[pole_dataclass.Pole]
        Optimized poles with real, positive weights to reconstruct F(w) via a sum over 
        Lorentzians
    """
    # Generate normalized target function F(w) and vector of frequencies w
    freq_vec_pos,target,initial_norm = lorentzian_reconstruction.prepare_target(cfg,input_poles)
    # Generate decomposition parameters defined by the initial guess. 
    decomp_parameters = lorentzian_reconstruction.poles_to_decomp_parameters(initial_guess)
    # Normalize weights of decomposition_parameters initial guess to the same as the 
    # target function
    decomp_parameters[LorPoleType.WEIGHT] /= initial_norm 
    decomp_parameters_next = None
    decomp_parameters_prev = None
    freq_vec_prev,target_prev,initial_norm_center = lorentzian_reconstruction.prepare_target(cfg,center_poles)
    decomp_parameters_center = lorentzian_reconstruction.poles_to_decomp_parameters(center_poles)
    decomp_parameters_center[LorPoleType.WEIGHT] /= initial_norm_center
    
    # Optimize initial_guess decomp_parameters via constrained linear least-squares
    decomp_parameters = lorentzian_optimization.generate_optimized_parameters(
        freq_vec_pos,
        target,
        decomp_parameters,
        time_vec,
        center_poles=decomp_parameters_center
    )
    
    # Convert optimized set of poles into universal dataclass form
    frequencies = (decomp_parameters[LorPoleType.WIDTH] + 
                   1j*decomp_parameters[LorPoleType.FREQUENCY])
    weights = decomp_parameters[LorPoleType.WEIGHT]*initial_norm
    poles = pole_dataclass.create_poles_from_arrays(frequencies,weights)

    return poles

def lorentzian_decomposition_w_esprit_input(
    cfg: Dict,
    input_poles: List[pole_dataclass.Pole],
    time_vec: NDArray[np.floating]
) -> List[pole_dataclass.Pole]:
    """
    Perform an explorative Lorentzian decomposition of a frequency-space function.

    Starting from a set of input poles (e.g., from ESPRIT or AAA), this function generates 
    the target F(w), iteratively estimates new Lorentzian positions and widths, and then 
    optimizes all parameters to reconstruct F(w) as a sum of Lorentzians with positive weights.
    
    Input
    ----------
    cfg: Dict
        Dictionary of universal parameters
    input_poles: List[pole_dataclass.Pole]
        Poles used to construct target function F(w) to decompose
    
    Output
    ------
    poles: List[pole_dataclass.Pole]
        Optimized poles with real, positive weights to reconstruct F(w) via a sum over 
        Lorentzians
    """
    # Generate normalized target function F(w) and vector of frequencies w
    freq_vec_pos,target,initial_norm = lorentzian_reconstruction.prepare_target(cfg,input_poles)
    estimate = np.zeros_like(target)
    residual = _generate_residual(target, estimate)
    # Estimate weight, width, and frequency of first Lorentzian
    decomp_parameters = lorentzian_initialization.estimate_weights_from_esprit(
        target,
        residual,
        freq_vec_pos,
        input_poles
    )
    decomp_parameters[LorPoleType.WEIGHT] /= initial_norm 
    # Fit via constrained least-squares to target function with constraint that the 
    # weight is real and positive
    decomp_parameters = lorentzian_optimization.generate_optimized_parameters(
        freq_vec_pos,
        target,
        decomp_parameters,
        time_vec
    )
    # Convert optimized set of poles into universal dataclass form
    frequencies = (decomp_parameters[LorPoleType.WIDTH] + 
                   1j*decomp_parameters[LorPoleType.FREQUENCY])
    weights = decomp_parameters[LorPoleType.WEIGHT]*initial_norm
    poles = pole_dataclass.create_poles_from_arrays(frequencies,weights)

    return poles
