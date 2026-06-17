import numpy as np
from source.utils.enums import LorPoleType
from typing import Dict , Optional
from numpy.typing import NDArray

def generate_width_estimate(dw,ind_omega,freq_vec,target):

    new_freq = freq_vec[ind_omega]
    dw = (freq_vec[ind_omega + 1] - freq_vec[ind_omega - 1]) / 2
    g0  = target[ind_omega]
    g1p = target[ind_omega+1]
    g1m = target[ind_omega-1]

    g_x  = (g1p - g1m) / (2*dw)
    g_xx = (g1p - 2*g0 + g1m) / (dw**2)

    D_pp = g_xx # - g_x) #/ (new_freq**2)
    eps_res = 1e-3 * np.max(np.abs(target))
    if g0 < eps_res or D_pp >= 0:
        new_width = 2 * dw
    else:
        new_width = max(np.sqrt(-2 * g0 / D_pp), 2 * dw)

    return new_width

def generate_pole_estimate(residual,freq_vec):

    residual_pos = np.maximum(residual, 0.0)
    if np.max(residual_pos) < 1e-12:
        # no positive residual left
        return None, None, None
    new_pole_ind = np.argmax(residual_pos)
    
    return freq_vec[new_pole_ind],new_pole_ind,residual_pos

def generate_weight_estimate(
    width,
    target,
    freq_vec,
    new_freq,
    new_width,
    residual_pos,
    Lambda = 4.0
    ):

    mask = np.abs(freq_vec - new_freq) < Lambda * new_width
    local_area = np.trapz(residual_pos[mask], freq_vec[mask])
    new_weight = max(local_area / np.pi, 1e-12)

    return new_weight

def estimate_new_pole(
    target,
    residual,
    freq_vec
    ):

    new_freq,new_pole_ind,residual_pos = generate_pole_estimate(
        residual,
        freq_vec
        )
    dw = np.log(freq_vec[1]) - np.log(freq_vec[0])

    new_width = generate_width_estimate(
        dw,
        new_pole_ind,
        freq_vec,
        target
        )

    new_weight = generate_weight_estimate(
        new_width,
        target,
        freq_vec,
        new_freq,
        new_width,
        residual_pos
        )

    if new_freq is not None:
        decomp_parameters_new_guess: Dict[LorPoleType, NDArray[np.floating]] = {
            LorPoleType.FREQUENCY : np.atleast_1d(new_freq),
            LorPoleType.WIDTH : np.atleast_1d(new_width),
            LorPoleType.WEIGHT : np.atleast_1d(new_weight)
    }

    return decomp_parameters_new_guess

def estimate_weights_from_esprit(
    target,
    residual,
    freq_vec,
    esprit_poles
    ):

    n_terms = len(esprit_poles)
    decomp_parameters_new_guess: Dict[LorPoleType, NDArray[np.floating]] = {
            LorPoleType.FREQUENCY : np.zeros(n_terms),
            LorPoleType.WIDTH : np.zeros(n_terms),
            LorPoleType.WEIGHT : np.zeros(n_terms)
    }

    for itr,pole in enumerate(esprit_poles):
        new_freq = pole.imag_part_freq
        new_width = pole.real_part_freq
        residual_pos = np.maximum(residual, 0.0)
        new_weight = generate_weight_estimate(
            new_width,
            target,
            freq_vec,
            new_freq,
            new_width,
            residual_pos
            )
        decomp_parameters_new_guess[LorPoleType.FREQUENCY][itr] = new_freq
        decomp_parameters_new_guess[LorPoleType.WIDTH][itr] = new_width
        decomp_parameters_new_guess[LorPoleType.WEIGHT][itr] = new_weight
        
    return decomp_parameters_new_guess