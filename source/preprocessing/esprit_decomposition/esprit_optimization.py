# source/preprocessing/lorentzian_decomposition/lorentzian_optimization.py
"""
For a given time vector and associated function, computes the ESPRIT
decomposition for a given number of terms in the decomposition and 
outputs the associated weights and frequencies.

Provides:
---------
- esprit_decomposition(function_integrand_exact,n_terms,L_param,
                        n_timesteps,dt):
    Given a function evaluated on a uniformly spaced interval, calculates and 
    returns the weights and frequencies from an ESPRIT decomposition of this function.

Usage:
------
from source.preprocessing import decomposition

dt,n_timesteps,uniform_time_vec = decomposition.uniform_time_parameter_generator(time_vec)
uniform_function_integrand,dt,n_timesteps = decomposition.function_integrand_w_uniform_time(time_vec,function_integrand)
weights,frequencies = decomposition.esprit_decomposition(time_vec,function_integrand)

n_polesotes:
------
- esprit_decomposition requires a function integrand evaluated on a uniform time grid
"""

import numpy as np
from scipy.optimize import (minimize)
from source.preprocessing.esprit_decomposition import esprit_reconstruction
from source.preprocessing.pole_dataclass import Pole
from typing import Dict , Optional , List
from numpy.typing import NDArray

# def generate_optimized_parameters(
#         time_vec: NDArray[np.floating],
#         target: NDArray[np.floating], 
#         initial_guess: List[Pole],
#         prev_poles: List[Pole]
# ):
    
#     """
#     Optimize the ESPRIT decomposition parameters: a_r,a_i,b_r,b_i.

#     Parameters
#     ----------
#     freq_vec : np.ndarray
#         Positive frequencies used for the fit.
#     target : np.ndarray
#         Target spectrum (normalized).
#     decomp_parameters : list of np.ndarray
#         Current decomposition parameters: [omega, width, weight].

#     Returns
#     -------
#     decomp_parameters_optimized : list of np.ndarray
#         Optimized [omega, width, weight], all positive arrays.
#     """

#     eps_freq = 1e-12   # protects log for small decay rates
#     eps_weight = 1e-10   # protects normalization near zero weight
#     dx = 0.1
#     lambda_smooth = 0.0
#     if prev_poles is not None:
#         lambda_smooth = 1e-5#compute_lambda_smooth(target,decomp_parameters,lambda0=1e-2)
#     # lambda_rep = 1e-3 * lambda_smooth
#     # lambda_poles = 0.0
#     # if center_poles is not None:
#     #     lambda_poles = 10 * lambda_smooth
#     # omega_floor = max(1e-12,freq_vec[1] - freq_vec[0])

#     n_poles = len(initial_guess)
#     freq_r_init = [initial_guess[i].real_part_freq for i in range(n_poles)]
#     freq_i_init = [initial_guess[i].imag_part_freq for i in range(n_poles)]
#     weight_r_init = [initial_guess[i].real_part_weight for i in range(n_poles)]
#     weight_i_init = [initial_guess[i].imag_part_weight for i in range(n_poles)]
    
#     freq_r_prev = np.array([prev_poles[i].real_part_freq for i in range(n_poles)])
#     freq_i_prev = np.array([prev_poles[i].imag_part_freq for i in range(n_poles)])
#     weight_r_prev = np.array([prev_poles[i].real_part_weight for i in range(n_poles)])
#     weight_i_prev = np.array([prev_poles[i].imag_part_weight for i in range(n_poles)])
    
#     # -------------------------------------------------------
#     # Flatten parameters into a single vector for the optimizer
#     # We use log-transform for width and weight to enforce positivity
#     # Omega can remain unconstrained
#     # params = [omega0..n_poles-1, log(width0..n_poles-1), log(weight0..n_poles-1)]
#     # -------------------------------------------------------
#     params_init = np.concatenate([
#         freq_r_init,
#         freq_i_init,
#         weight_r_init,
#         weight_i_init
#     ])

#     def objective(params):
#         # Unpack
#         freq_r_opt = params[0:n_poles]
#         freq_i_opt  = params[n_poles:2*n_poles]
#         weight_r_opt = params[2*n_poles:3*n_poles]
#         weight_i_opt = params[3*n_poles:4*n_poles]

#         # Log-smoothness for decay rates (Re(freq))
#         log_freq_r_opt  = np.log(freq_r_opt + eps_freq)
#         log_freq_r_prev = np.log(freq_r_prev + eps_freq)

#         penalty_freq_r = np.sum(((log_freq_r_opt - log_freq_r_prev)/dx)**2)

#         # Linear smoothness for oscillation frequencies (Im(freq))
#         penalty_freq_i = 1e-1 * np.sum(
#             ((freq_i_opt - freq_i_prev)/dx)**2 * (freq_i_prev**2 / (freq_i_prev**2 + eps_freq))
#         )

#         # Normalized smoothness for weights (complex)
#         penalty_weight = 1e-5*np.sum(
#             (((weight_r_opt - weight_r_prev)/dx)**2 +
#             ((weight_i_opt - weight_i_prev)/dx)**2)
#             / (weight_r_prev**2 + weight_i_prev**2 + eps_weight)
#         )

#         gamma_prev = np.sum(np.real((weight_r_prev + 1j*weight_i_prev) * np.exp(
#             -(freq_r_prev + 1j*freq_i_prev) * time_vec[:,None])), axis=1)
#         gamma_opt  = np.sum(np.real((weight_r_opt + 1j*weight_i_opt) * np.exp(-
#             (freq_r_opt + 1j*freq_i_opt) * time_vec[:,None])), axis=1)
#         penalty_gamma = np.mean((gamma_opt - gamma_prev)**2)

#         penalty = penalty_freq_r + penalty_freq_i + penalty_weight + penalty_gamma
#         penalty /= n_poles
#         # penalty_poles = 0.0
#         # if center_poles is not n_polesone:
#         #     log_omega_center  = np.log(np.abs(omega_center) + omega_floor)
#         #     log_width_center  = np.log(width_center)
#         #     log_weight_center = np.log(weight_center)
#         #     penalty_poles += np.sum(
#         #         (log_omega_opt - log_omega_center)**2 +
#         #         (log_width_opt - log_width_center)**2
#         #         )
        
#         # Reconstruct spectrum
#         estimate = np.zeros_like(time_vec)
#         for i in range(n_poles):
#             freq = freq_r_opt[i] + 1j*freq_i_opt[i]
#             weight = weight_r_opt[i] + 1j*weight_i_opt[i]
#             estimate += np.real(weight * np.exp(-freq * time_vec))

#         # Weighted squared error (can adjust weights if needed)
#         err = target - estimate
#         return (np.mean(err**2) + lambda_smooth * penalty)
#         # + lambda_poles * penalty)

#     # -------------------------------------------------------
#     # Run optimizer (can tune method and options)
#     # -------------------------------------------------------
#     dt = time_vec[1] - time_vec[0]
#     bounds = (
#         [(0.0, None)] * n_poles + 
#         [(None, None)] * (3 * n_poles)
#     )
#     res = minimize(
#         objective,
#         params_init,
#         method='L-BFGS-B',
#         bounds=bounds,
#         options={'maxiter': 2000, 'ftol': 1e-12}
#     )

#     # -------------------------------------------------------
#     # Unpack optimized parameters
#     # -------------------------------------------------------
#     params_opt = res.x
#     freq_opt = params_opt[0:n_poles] + 1j*params_opt[n_poles:2*n_poles]
#     weight_opt = params_opt[2*n_poles:3*n_poles] + 1j*params_opt[3*n_poles:4*n_poles]
    
#     return weight_opt,freq_opt

def generate_optimized_parameters(
        time_vec: NDArray[np.floating],
        target: NDArray[np.floating], 
        initial_guess: List[Pole],
        prev_poles: Optional[List[Pole]] = None
):
    """
    Optimize ESPRIT decomposition parameters (weights and complex frequencies) 
    with smooth evolution across x, handling complex -> real transitions naturally.

    Parameters
    ----------
    time_vec : NDArray
        Uniformly spaced time points.
    target : NDArray
        Target function (e.g., gamma(x,t)) at this x.
    initial_guess : List[Pole]
        Poles from current ESPRIT decomposition at this x.
    prev_poles : Optional[List[Pole]]
        Poles from previous x-coordinate, for smoothness.

    Returns
    -------
    weight_opt : NDArray[complex]
        Optimized complex weights.
    freq_opt : NDArray[complex]
        Optimized complex frequencies.
    """

    eps_freq = 1e-12
    eps_weight = 1e-10
    dx = 0.1  # step in x, used for normalization in penalties
    n_poles = len(initial_guess)

    # Extract initial parameters
    freq_r_init = np.array([p.real_part_freq for p in initial_guess])
    freq_i_init = np.array([p.imag_part_freq for p in initial_guess])
    weight_r_init = np.array([p.real_part_weight for p in initial_guess])
    weight_i_init = np.array([p.imag_part_weight for p in initial_guess])

    # Flatten for optimizer
    params_init = np.concatenate([freq_r_init, freq_i_init, weight_r_init, weight_i_init])

    # Prepare previous pole arrays if available
    if prev_poles is not None:
        freq_r_prev = np.array([p.real_part_freq for p in prev_poles])
        freq_i_prev = np.array([p.imag_part_freq for p in prev_poles])
        weight_r_prev = np.array([p.real_part_weight for p in prev_poles])
        weight_i_prev = np.array([p.imag_part_weight for p in prev_poles])
    else:
        freq_r_prev = freq_r_init
        freq_i_prev = freq_i_init
        weight_r_prev = weight_r_init
        weight_i_prev = weight_i_init

    def objective(params):
        # Unpack
        freq_r_opt = params[0:n_poles]
        freq_i_opt = params[n_poles:2*n_poles]
        weight_r_opt = params[2*n_poles:3*n_poles]
        weight_i_opt = params[3*n_poles:4*n_poles]
        weight_opt = weight_r_opt + 1j*weight_i_opt
        freq_opt = freq_r_opt + 1j*freq_i_opt

        # ------------------------
        # Smoothness penalties
        # ------------------------
        penalty = 0.0

        # Decay rate smoothness (log)
        freq_r_safe = np.maximum(freq_r_opt, eps_freq)
        log_freq_r_opt = np.log(freq_r_safe)
        log_freq_r_prev = np.log(np.maximum(freq_r_prev, eps_freq))
        penalty += np.sum(((log_freq_r_opt - log_freq_r_prev)/dx)**2)

        # Imaginary part smoothness, adaptive: reduces as mode becomes overdamped
        adapt = freq_i_prev**2 / (freq_i_prev**2 + eps_freq)
        penalty += 1e-1 * np.sum(((freq_i_opt - freq_i_prev)/dx)**2 * adapt)

        # Weight smoothness (normalized)
        denom = weight_r_prev**2 + weight_i_prev**2 + eps_weight
        penalty += 1e-5 * np.sum(((weight_r_opt - weight_r_prev)/dx)**2 / denom)
        penalty += 1e-5 * np.sum(((weight_i_opt - weight_i_prev)/dx)**2 / denom)

        # Smoothness on total reconstructed function gamma(x,t)
        gamma_prev = np.sum((weight_r_prev + 1j*weight_i_prev)[:,None] *
                            np.exp(-(freq_r_prev + 1j*freq_i_prev)[:,None]*time_vec), axis=0)
        gamma_opt = np.sum(weight_opt[:,None] * np.exp(-freq_opt[:,None]*time_vec), axis=0)
        penalty += 1e-4 * np.mean(np.abs(gamma_opt - gamma_prev)**2)

        # ------------------------
        # Reconstruction error
        # ------------------------
        estimate = np.real(gamma_opt)
        err = target - estimate

        return np.mean(err**2) + penalty

    # Bounds: ensure decay rates >= 0
    bounds = [(0.0, None)]*n_poles + [(None, None)]*(3*n_poles)

    # Run optimizer
    res = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter':2000, 'ftol':1e-12}
    )

    # Unpack optimized parameters
    params_opt = res.x
    freq_opt = params_opt[0:n_poles] + 1j*params_opt[n_poles:2*n_poles]
    weight_opt = params_opt[2*n_poles:3*n_poles] + 1j*params_opt[3*n_poles:4*n_poles]

    return weight_opt, freq_opt