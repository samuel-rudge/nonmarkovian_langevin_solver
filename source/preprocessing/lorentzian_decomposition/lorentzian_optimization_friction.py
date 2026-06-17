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
from typing import Dict , Optional
from numpy.typing import NDArray

def generate_optimized_parameters_signed_weights(
        freq_vec: NDArray[np.floating],
        target: NDArray[np.floating],
        decomp_parameters: Dict[LorPoleType, NDArray[np.floating]],
        time_vec: NDArray[np.floating],
        center_poles: Optional[Dict[LorPoleType, NDArray[np.floating]]] = None
) -> Dict[LorPoleType, NDArray[np.floating]]:
    """
    Optimize Lorentzian decomposition with unconstrained (signed) weights.
    Smoothness is enforced only at the spectrum level, not on individual weights.
    """

    max_width = 1 / (time_vec[1] - time_vec[0])
    min_width = 1 / np.max(time_vec)

    lambda_smooth = 0.0
    if center_poles is not None:
        lambda_smooth = 1e-2

    lambda_poles = 0.0
    if center_poles is not None:
        lambda_poles = 10 * lambda_smooth

    omega_floor = max(1e-12, freq_vec[1] - freq_vec[0])

    omega_init  = decomp_parameters[LorPoleType.FREQUENCY]
    width_init  = decomp_parameters[LorPoleType.WIDTH]
    weight_init = decomp_parameters[LorPoleType.WEIGHT]
    N = len(omega_init)

    if center_poles is not None:
        omega_center  = center_poles[LorPoleType.FREQUENCY]
        width_center  = center_poles[LorPoleType.WIDTH]

    # -------------------------------------------------------
    # Parameter vector:
    # [omega_0..N-1, log(width_0)..log(width_N-1), weight_0..weight_N-1]
    # -------------------------------------------------------
    params_init = np.concatenate([
        omega_init,
        np.log(width_init),
        weight_init
    ])

    # Precompute previous spectrum (for smoothness)
    def reconstruct_spectrum(omega, width, weight):
        spec = np.zeros_like(freq_vec)
        for i in range(N):
            spec += weight[i] * (
                width[i] / ((freq_vec - omega[i])**2 + width[i]**2) +
                width[i] / ((freq_vec + omega[i])**2 + width[i]**2)
            )
        return spec

    spectrum_prev = reconstruct_spectrum(
        omega_init, width_init, weight_init
    )

    def objective(params):
        omega_opt  = params[0:N]
        width_opt  = np.exp(params[N:2*N])
        weight_opt = params[2*N:3*N]

        # Reconstruct spectrum
        spectrum_opt = reconstruct_spectrum(
            omega_opt, width_opt, weight_opt
        )

        # ---------------------------------------------------
        # Data fidelity: linear least squares in spectrum
        # ---------------------------------------------------
        err = target - spectrum_opt
        loss = np.mean(err**2)

        # ---------------------------------------------------
        # Smoothness: spectrum-level only
        # ---------------------------------------------------
        penalty = lambda_smooth * np.mean(
            (spectrum_opt - spectrum_prev)**2
        )

        # ---------------------------------------------------
        # Center anchoring: pole geometry only
        # ---------------------------------------------------
        penalty_poles = 0.0
        if center_poles is not None:
            log_omega_opt   = np.log(np.abs(omega_opt) + omega_floor)
            log_width_opt   = np.log(width_opt)
            log_omega_c     = np.log(np.abs(omega_center) + omega_floor)
            log_width_c     = np.log(width_center)

            penalty_poles = lambda_poles * np.mean(
                (log_omega_opt - log_omega_c)**2 +
                (log_width_opt - log_width_c)**2
            )

        return loss + penalty + penalty_poles

    # -------------------------------------------------------
    # Bounds
    # -------------------------------------------------------
    omega_bounds      = [(freq_vec.min(), freq_vec.max())] * N
    log_width_bounds  = [(np.log(min_width), np.log(max_width))] * N
    weight_bounds     = [(None, None)] * N

    bounds = omega_bounds + log_width_bounds + weight_bounds

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
    omega_opt  = params_opt[0:N]
    width_opt  = np.exp(params_opt[N:2*N])
    weight_opt = params_opt[2*N:3*N]

    return {
        LorPoleType.FREQUENCY: omega_opt,
        LorPoleType.WIDTH: width_opt,
        LorPoleType.WEIGHT: weight_opt
    }