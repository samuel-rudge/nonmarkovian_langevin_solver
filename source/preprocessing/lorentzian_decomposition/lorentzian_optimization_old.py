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

Notes:
------
- esprit_decomposition requires a function integrand evaluated on a uniform time grid
"""

import numpy as np
from scipy.optimize import (minimize)
from source.utils.enums import LorPoleType
from typing import Dict
from numpy.typing import NDArray

def compute_lambda_smooth(target,decomp_parameters,lambda0=1e-3):

    # Spectral error scale
    E_spec = np.mean(np.log(target + 1e-12)**2)

    # Pole scale
    omega = decomp_parameters[LorPoleType.FREQUENCY]
    width = decomp_parameters[LorPoleType.WIDTH]
    weight = decomp_parameters[LorPoleType.WEIGHT]
    S_scale = np.mean(omega**2 + width**2 + weight**2)

    lambda_smooth = lambda0 * E_spec / (S_scale + 1e-12)
    return lambda_smooth

def generate_optimized_parameters(
        freq_vec: NDArray[np.floating], 
        target: NDArray[np.floating], 
        decomp_parameters: Dict[LorPoleType, NDArray[np.floating]],
        time_vec: NDArray[np.floating],
        next_poles=None,
        prev_poles=None
) -> Dict[LorPoleType, NDArray[np.floating]]:
    """
    Optimize the Lorentzian decomposition parameters: omega, width, weight.

    Parameters
    ----------
    freq_vec : np.ndarray
        Positive frequencies used for the fit.
    target : np.ndarray
        Target spectrum (normalized).
    decomp_parameters : list of np.ndarray
        Current decomposition parameters: [omega, width, weight].

    Returns
    -------
    decomp_parameters_optimized : list of np.ndarray
        Optimized [omega, width, weight], all positive arrays.
    """

    max_width = 1/(time_vec[1] - time_vec[0])
    min_width = 1/(np.max(time_vec))
    lambda_smooth = 1e-2#compute_lambda_smooth(target,decomp_parameters,lambda0=1e-2)
    lambda_rep = 1e-3 * lambda_smooth
    lambda_2_smooth = 0.05 * lambda_smooth
    omega_floor = max(1e-12,freq_vec[1] - freq_vec[0])

    omega_init = decomp_parameters[LorPoleType.FREQUENCY]
    width_init = decomp_parameters[LorPoleType.WIDTH]
    weight_init = decomp_parameters[LorPoleType.WEIGHT]
    N = len(omega_init)
    if next_poles is not None and prev_poles is not None:
        omega_next = next_poles[LorPoleType.FREQUENCY]
        width_next = next_poles[LorPoleType.WIDTH]
        weight_next = next_poles[LorPoleType.WEIGHT]
        omega_prev = prev_poles[LorPoleType.FREQUENCY]
        width_prev = prev_poles[LorPoleType.WIDTH]
        weight_prev = prev_poles[LorPoleType.WEIGHT]
    else:
        omega_prev = decomp_parameters[LorPoleType.FREQUENCY]
        width_prev = decomp_parameters[LorPoleType.WIDTH]
        weight_prev = decomp_parameters[LorPoleType.WEIGHT]
    # -------------------------------------------------------
    # Flatten parameters into a single vector for the optimizer
    # We use log-transform for width and weight to enforce positivity
    # Omega can remain unconstrained
    # params = [omega0..N-1, log(width0..N-1), log(weight0..N-1)]
    # -------------------------------------------------------
    params_init = np.concatenate([
        omega_init,
        np.log(width_init),
        np.log(weight_init)
    ])

    def objective(params):
        # Unpack
        omega_opt = params[0:N]
        width_opt  = np.exp(params[N:2*N])#np.exp(np.clip(params[N:2*N], -20, 20))
        weight_opt = np.exp(params[2*N:3*N])#np.exp(np.clip(params[2*N:3*N], -20, 20))

        # Construct penalty
        log_omega_prev  = np.log(np.abs(omega_prev) + omega_floor)
        log_width_prev  = np.log(width_prev)
        log_weight_prev = np.log(weight_prev)
        
        log_omega_opt  = np.log(np.abs(omega_opt) + omega_floor)
        log_width_opt  = np.log(width_opt)
        log_weight_opt = np.log(weight_opt)
        penalty = np.sum(
            (log_omega_opt - log_omega_prev)**2 +
            (log_width_opt - log_width_prev)**2 +
            (log_weight_opt - log_weight_prev)**2
        )
        penalty_2 = 0.0
        if next_poles is not None:
            log_omega_next  = np.log(np.abs(omega_next) + omega_floor)
            log_width_next  = np.log(width_next)
            log_weight_next = np.log(weight_next)
            penalty += np.sum(
                (log_omega_opt - log_omega_next)**2 +
                (log_width_opt - log_width_next)**2 +
                (log_weight_opt - log_weight_next)**2
                )
            # Add second-order (derivative) penalty
            penalty_2 += np.sum(
                (log_omega_next - 2*log_omega_opt + log_omega_prev)**2 +
                (log_width_next - 2*log_width_opt + log_width_prev)**2 +
                (log_weight_next - 2*log_weight_opt + log_weight_prev)**2
                )
        
        # Pole repulsion
        repulsion = 0.0
        delta = 0.05 * (freq_vec.max() - freq_vec.min()) / N
        for i in range(N):
            for j in range(i+1, N):
                repulsion += 1.0 / ((omega_opt[i] - omega_opt[j])**2 + delta**2)
        if N > 1:
            repulsion /= N*(N-1)/2

        # Reconstruct spectrum
        estimate = np.zeros_like(freq_vec)
        for i in range(N):
            estimate += weight_opt[i] * (
                width_opt[i] / ((freq_vec - omega_opt[i])**2 + width_opt[i]**2) +
                width_opt[i] / ((freq_vec + omega_opt[i])**2 + width_opt[i]**2)
            )
        # err = target - estimate

        # Weighted squared error (can adjust weights if needed)
        # weights = 1 / (target + 1e-12)  # boost relative error in small peaks
        # err_weighted = weights * (target - estimate)
        err_log = np.log(target + 1e-12) - np.log(estimate + 1e-12)
        # return np.mean(err_weighted**2)
        if next_poles is None:
            return (np.mean(err_log**2) + lambda_smooth * penalty + lambda_rep * repulsion)
        else:
            return (np.mean(err_log**2) + lambda_2_smooth * penalty_2 + lambda_rep * repulsion)
        
        # return (np.mean(err_log**2) + lambda_smooth * penalty + 
        #         lambda_2_smooth * penalty_2 + lambda_rep * repulsion)

    # -------------------------------------------------------
    # Run optimizer (can tune method and options)
    # -------------------------------------------------------
    omega_bounds = omega_bounds = [(freq_vec.min(), freq_vec.max())] * N
    log_width_bounds  = [(np.log(min_width), np.log(max_width))] * N
    log_weight_bounds = [(np.log(1e-4), np.log(1e1))] * N
    bounds = omega_bounds + log_width_bounds + log_weight_bounds
    res = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12}
    )

    # -------------------------------------------------------
    # Unpack optimized parameters
    # -------------------------------------------------------
    params_opt = res.x
    omega_opt = params_opt[0:N]
    width_opt  = np.exp(params_opt[N:2*N])#np.exp(np.clip(params_opt[N:2*N], -5, 2))
    weight_opt = np.exp(params_opt[2*N:3*N])#np.exp(np.clip(params_opt[2*N:3*N], -5, 2))

    decomp_parameters_optimized: Dict[LorPoleType, NDArray[np.floating]] = {
        LorPoleType.FREQUENCY : omega_opt,
        LorPoleType.WIDTH : width_opt,
        LorPoleType.WEIGHT : weight_opt
    }

    return decomp_parameters_optimized
