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

# -----------------------------
# Helper: pair conjugate poles
# -----------------------------
def pair_conjugate_poles(poles: List[Pole]) -> List[Pole]:
    """
    Reorder a list of poles into explicit conjugate pairs.
    Each pair (b, conj(b)) will occupy adjacent slots,
    with the positive imaginary part first.
    """
    n_poles = len(poles)
    if n_poles % 2 != 0:
        raise ValueError("Number of poles must be even to form conjugate pairs.")

    paired = [False] * n_poles
    new_order = []

    for i, p in enumerate(poles):
        if paired[i]:
            continue

        min_dist = np.inf
        min_j = -1
        for j, q in enumerate(poles):
            if i == j or paired[j]:
                continue
            # Conjugacy distance metric
            dist = abs(p.real_part_freq - q.real_part_freq) + abs(p.imag_part_freq + q.imag_part_freq)
            if dist < min_dist:
                min_dist = dist
                min_j = j

        # If no match found (imag ≈ 0), treat as self-conjugate
        if min_j == -1 or np.isclose(poles[min_j].imag_part_freq, 0.0, atol=1e-12):
            new_order.append(p)
            new_order.append(p)  # duplicate for latent mode
            paired[i] = True
        else:
            # Positive imaginary first
            if p.imag_part_freq >= 0:
                new_order.append(p)
                new_order.append(poles[min_j])
            else:
                new_order.append(poles[min_j])
                new_order.append(p)
            paired[i] = True
            paired[min_j] = True

    # Remove duplicates for real poles
    new_order = [p for idx, p in enumerate(new_order) if not (np.isclose(p.imag_part_freq, 0.0) and idx % 2 == 1)]

    return new_order


def generate_optimized_parameters(
    time_vec: NDArray[np.floating],
    target: NDArray[np.floating], 
    initial_guess: List[Pole],
    prev_poles: Optional[List[Pole]] = None
):
    """
    Optimize the ESPRIT decomposition parameters using latent conjugate-pair coordinates.

    Parameters
    ----------
    time_vec : np.ndarray
        Uniform time vector.
    target : np.ndarray
        Function to fit (e.g., friction kernel at a single x).
    initial_guess : List[Pole]
        ESPRIT poles for this x, already tracked relative to previous x.
    prev_poles : List[Pole], optional
        Poles from previous x for smoothness regularization.

    Returns
    -------
    weight_opt : np.ndarray[complex]
        Optimized pole weights.
    freq_opt : np.ndarray[complex]
        Optimized pole frequencies (complex).
    """

    omega_0 = 1/(np.max(time_vec) - np.min(time_vec))
    eps_weight = 1e-10
    dx = 0.1
    lambda_smooth = 0.0
    n_poles = len(initial_guess)
    n_modes = n_poles // 2  # assumes even number
    print(n_poles)

    # -----------------------------
    # Step 0: enforce conjugate-pair adjacency
    # -----------------------------
    initial_guess = pair_conjugate_poles(initial_guess)
    if prev_poles is not None:
        prev_poles = pair_conjugate_poles(prev_poles)

    # -------------------------------------------------------
    # Step 1: Convert ESPRIT poles to latent conjugate-pair coordinates
    # -------------------------------------------------------
    alpha_init = np.zeros(n_modes)
    omega_init = np.zeros(n_modes)
    C_init = np.zeros(n_modes)
    S_init = np.zeros(n_modes)

    for j in range(n_modes):
        k = 2*j
        b_k_real = initial_guess[k].real_part_freq
        b_k_imag = initial_guess[k].imag_part_freq
        b_kp1_real = initial_guess[k+1].real_part_freq
        b_kp1_imag = initial_guess[k+1].imag_part_freq
        w_k_real = initial_guess[k].real_part_weight
        w_k_imag = initial_guess[k].imag_part_weight
        w_kp1_real = initial_guess[k+1].real_part_weight
        w_kp1_imag = initial_guess[k+1].imag_part_weight

        alpha_init[j] = 0.5 * (b_k_real + b_kp1_real)
        omega_init[j] = 0.5 * abs(b_k_imag - b_kp1_imag)
        C_init[j] = w_k_real + w_kp1_real
        S_init[j] = w_k_imag - w_kp1_imag

    params_init = np.concatenate([alpha_init, omega_init, C_init, S_init])

    # -------------------------------------------------------
    # Step 2: Convert previous-x poles to latent coordinates if available
    # -------------------------------------------------------
    if prev_poles is not None:
        alpha_prev = np.zeros(n_modes)
        omega_prev = np.zeros(n_modes)
        C_prev = np.zeros(n_modes)
        S_prev = np.zeros(n_modes)

        for j in range(n_modes):
            k = 2*j
            b_k_real = prev_poles[k].real_part_freq
            b_k_imag = prev_poles[k].imag_part_freq
            b_kp1_real = prev_poles[k+1].real_part_freq
            b_kp1_imag = prev_poles[k+1].imag_part_freq
            w_k_real = prev_poles[k].real_part_weight
            w_k_imag = prev_poles[k].imag_part_weight
            w_kp1_real = prev_poles[k+1].real_part_weight
            w_kp1_imag = prev_poles[k+1].imag_part_weight

            alpha_prev[j] = 0.5 * (b_k_real + b_kp1_real)
            omega_prev[j] = 0.5 * abs(b_k_imag - b_kp1_imag)
            C_prev[j] = w_k_real + w_kp1_real
            S_prev[j] = w_k_imag - w_kp1_imag

        lambda_smooth_freq = 1e-5  # can adjust
        lambda_smooth_weight = lambda_smooth_freq / 1e3
        lambda_omega = 1e-4#5*lambda_smooth
    else:
        alpha_prev = omega_prev = C_prev = S_prev = None

    # -------------------------------------------------------
    # Step 3: Objective function in latent coordinates
    # -------------------------------------------------------
    def objective(params):
        alpha_opt = params[0:n_modes]
        omega_opt = params[n_modes:2*n_modes]
        C_opt = params[2*n_modes:3*n_modes]
        S_opt = params[3*n_modes:4*n_modes]

        # Reconstruction of function
        estimate = np.zeros_like(time_vec)
        for j in range(n_modes):
            estimate += np.exp(-alpha_opt[j] * time_vec) * (
                C_opt[j]*np.cos(omega_opt[j]*time_vec) + S_opt[j]*np.sin(omega_opt[j]*time_vec)
            )

        # Weighted squared error
        err = target - estimate
        loss = np.mean(err**2)

        # Smoothness penalty
        if prev_poles is not None:
            penalty_alpha = lambda_smooth_freq * np.sum(((alpha_opt - alpha_prev)/dx)**2)
            penalty_omega = lambda_smooth_freq * np.sum(((omega_opt - omega_prev)/dx)**2)
            penalty_C = lambda_smooth_weight * np.sum(((C_opt - C_prev)/dx)**2 / (C_prev**2 + S_prev**2 + eps_weight))
            penalty_S =lambda_smooth_weight * np.sum(((S_opt - S_prev)/dx)**2 / (C_prev**2 + S_prev**2 + eps_weight))
            loss += (penalty_alpha + penalty_omega + penalty_C + penalty_S)/n_modes
            amp_sq = C_opt**2 + S_opt**2
            loss += lambda_omega * np.sum(amp_sq * np.log(omega_opt**2 + omega_0**2)) / n_modes
            #loss += lambda_omega * np.sum(np.log(omega_opt**2 + omega_0**2)) / n_modes

        return loss

    # -------------------------------------------------------
    # Step 4: Run optimizer
    # -------------------------------------------------------
    bounds = [(0.0, None)]*n_modes + [(None, None)]*3*n_modes  # alpha >=0, rest free
    res = minimize(
        objective,
        params_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12}
    )

    # -------------------------------------------------------
    # Step 5: Convert optimized latent variables back to complex poles
    # -------------------------------------------------------
    params_opt = res.x
    alpha_opt = params_opt[0:n_modes]
    omega_opt = params_opt[n_modes:2*n_modes]
    C_opt = params_opt[2*n_modes:3*n_modes]
    S_opt = params_opt[3*n_modes:4*n_modes]
    for j in range(n_modes):
        if omega_opt[j] < 0:
            omega_opt[j] = -omega_opt[j]
            S_opt[j] = -np.abs(S_opt[j])

    freq_opt = np.zeros(n_poles, dtype=complex)
    weight_opt = np.zeros(n_poles, dtype=complex)
    for j in range(n_modes):
        k = 2*j
        freq_opt[k] = alpha_opt[j] + 1j*omega_opt[j]
        freq_opt[k+1] = alpha_opt[j] - 1j*omega_opt[j]
        weight_opt[k] = 0.5 * (C_opt[j] + 1j*S_opt[j])
        weight_opt[k+1] = 0.5 * (C_opt[j] - 1j*S_opt[j])

    return weight_opt, freq_opt
