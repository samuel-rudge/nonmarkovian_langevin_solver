# source/preprocessing/pole_management.py
"""
Contains various functions that manage friction and corrfunc poles after decomposition.
This includes converting them to arrays to be read by the later Langevin implementation, 
a tracking process that ensures each pole is ordered correctly at each coordinate, and 
interpolation procedures.

Provides:
---------
- poles_to_arrays(
        x_coordinate_vec : NDArray[np.floating],
        poles_arr : List[List[pole_dataclass.Pole]]
) -> Dict[str, NDArray[np.floating]]:
    Given a set of poles defined over a coordinate grid, returns them in an array form that 
    can be read by the Langevin implementation.
- pole_interpolator(x_coordinate,,arr_to_interp):
    Given a set of poles defined over a coordinate grid, interpolates them with cubic splines 
    and returns a list of n_poles functions. 
- interp_poles_to_arrays(pole_input):
    Given a set of interpolated pole functions, returns them in array readable by the 
    Langevin algorithm. 
- pole_tracker(poles_untracked):
    Given a set of untracked poles defined over a coordinate grid, uses a distance measure to 
    ensure that at each x, the same pole index is used for each pole (i.e. that Pole 1 always
    refers to the same pole as x changes).
    
Usage:
------
from source.preprocessing import pole_management
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import linear_sum_assignment
from source.preprocessing import pole_dataclass
from typing import List , Dict, Tuple
from numpy.typing import NDArray

def poles_to_arrays(
        x_coordinate_vec : NDArray[np.floating],
        poles_arr : List[List[pole_dataclass.Pole]]
) -> Dict[str, NDArray[np.floating]]:
    """
    Takes a grid of x-coordinates and a set of poles (weights and frequencies) defined at each
    x-coordinate, and converts them into a dictionary containing arrays of the weights and
    frequencies of each pole at each x, which is readable by the Langevin propagation.
    
    Input:
    ------
    x_coordinate_vec : NDArray[np.floating],
        Grid of x-coordinates over which the decomposition is defined
    poles_arr : List[List[pole_dataclass.Pole]]
        Array of poles obtained from ESPRIT or Lorentzian decomposition at each x-value
    
    Output: 
    -------
    pole_array: Dict[str, NDArray[np.floating]]
        Dictionary of pole values at each x. For example, if one wants to access the real part 
        of the frequency of pole 1, one would use pole_array["freq_real"][0], which would 
    """
    # Calculate length of coordinate grid and number of poles at each x-value
    N_x_coordinate = x_coordinate_vec.shape[0]
    n_terms = len(poles_arr[0])
    # Set up weights and frequencies arrays
    freq_real = np.zeros((N_x_coordinate,n_terms),dtype=float)
    freq_imag = np.zeros((N_x_coordinate,n_terms),dtype=float)
    weights_real = np.zeros((N_x_coordinate,n_terms),dtype=float)
    weights_imag = np.zeros((N_x_coordinate,n_terms),dtype=float)
    # Loop through x-values and poles at each x to fill arrays
    for i,poles in enumerate(poles_arr):
        for j, p in enumerate(poles):
            freq_real[i, j] = p.real_part_freq
            freq_imag[i, j] = p.imag_part_freq
            weights_real[i, j] = p.real_part_weight
            weights_imag[i, j] = p.imag_part_weight

    # Return dictionary
    return {
        "x": x_coordinate_vec,
        "freq_real": freq_real,
        "freq_imag": freq_imag,
        "weight_real": weights_real,
        "weight_imag": weights_imag
    }

def pole_interpolator(
        x_coordinate: NDArray[np.floating],
        arr_to_interp: NDArray[np.floating]
):

    # Calculate number of poles at each x-value
    n_terms = arr_to_interp.shape[1]
    
    return [CubicSpline(x_coordinate,arr_to_interp[:,itrp]) for itrp in range(n_terms)]

def interp_poles_to_arr(pole_input):

    x_coordinate_vec = pole_input["x"]
    freq_real = pole_input["freq_real"]
    freq_imag = pole_input["freq_imag"]
    weights_real = pole_input["weight_real"]
    weights_imag = pole_input["weight_imag"]

    freq_real_interp = pole_interpolator(x_coordinate_vec, freq_real)
    freq_imag_interp = pole_interpolator(x_coordinate_vec, freq_imag)
    weights_real_interp = pole_interpolator(x_coordinate_vec, weights_real)
    weights_imag_interp = pole_interpolator(x_coordinate_vec, weights_imag)

    return {
        "freq_real": freq_real_interp,
        "freq_imag": freq_imag_interp,
        "weight_real": weights_real_interp,
        "weight_imag": weights_imag_interp
    }

def pole_tracker(
    poles_untracked: List[List[pole_dataclass.Pole]]
) -> List[List[pole_dataclass.Pole]]:
    """
    Takes untracked poles and reorders them such they evolve (mostly) smoothly as a function
    of x. For example, since ESPRIT interpolates each coordinate independently, at two
    neighbouring x-values, it may arbitrarily decide that what it calls Pole 1 and Pole 2
    are reversed. Physically, however, we would expect that Pole 1 evolves smoothly as a 
    function of x, so this functions goes through and reorders the poles such that this is 
    satisfied (using a distance metric)
    
    Input
    -----
    poles_untracked: List[pole_dataclass.Pole]

    Output
    ------
    poles_tracked: List[pole_dataclass.Pole]
        
    """

    poles_tracked = [list(poles_untracked[0])]
    for itrx,poles in enumerate(poles_untracked[1:], start=1):
        prev_poles = poles_tracked[-1]
        curr_poles = list(poles_untracked[itrx])

        assigned = []
        new_order = []
        for p_prev in prev_poles:
            # Distance matrix (n_poles x n_poles)
            dist_pole = [np.sqrt(
                (p_prev.real_part_freq - p_curr.real_part_freq)**2 + 
                (p_prev.imag_part_freq - p_curr.imag_part_freq)**2
                ) for p_curr in curr_poles]
            idx_min = np.argmin(dist_pole)
        
            # Assign this closest pole
            new_order.append(curr_poles[idx_min])

            # Remove assigned pole from current pool
            curr_poles.pop(idx_min)
        # # Hungarian algorithm = optimal assignment
        # row_ind, col_ind = linear_sum_assignment(dist_pole)

        # # Reorder curr_pole according to the assignment
        poles_tracked.append(new_order)#[itrx] = [poles[itrp] for itrp in col_ind]
        # Reset for next loop
        # prev_pole = np.copy(curr_pole)

    return poles_tracked


def track_poles_pair(
    prev_poles: List[pole_dataclass.Pole],
    curr_poles: List[pole_dataclass.Pole]
) -> List[pole_dataclass.Pole]:
    """
    Takes untracked poles and reorders them such they evolve (mostly) smoothly as a function
    of x. For example, since ESPRIT interpolates each coordinate independently, at two
    neighbouring x-values, it may arbitrarily decide that what it calls Pole 1 and Pole 2
    are reversed. Physically, however, we would expect that Pole 1 evolves smoothly as a 
    function of x, so this functions goes through and reorders the poles such that this is 
    satisfied (using a distance metric)
    
    Input
    -----
    poles_untracked: List[pole_dataclass.Pole]

    Output
    ------
    poles_tracked: List[pole_dataclass.Pole]
        
    """

    # curr_poles = list(curr_poles)
    # new_order = []
    # for p_prev in prev_poles:
    #     dist_pole = [np.sqrt(
    #         (p_prev.real_part_freq - p_curr.real_part_freq)**2 + 
    #         (p_prev.imag_part_freq - p_curr.imag_part_freq)**2
    #         ) for p_curr in curr_poles]
    #     idx_min = np.argmin(dist_pole)
    #     # Assign this closest pole
    #     new_order.append(curr_poles.pop(idx_min))

    n_poles = len(prev_poles)
    dist_matrix = np.zeros((n_poles, n_poles), dtype=float)
    for i, p_prev in enumerate(prev_poles):
        for j, p_curr in enumerate(curr_poles):
            # Include weight magnitude and sign (imag part)
            freq_dist = (p_prev.real_part_freq - p_curr.real_part_freq) ** 2 + \
                        (p_prev.imag_part_freq - p_curr.imag_part_freq) ** 2
            weight_dist = abs((p_prev.real_part_weight + 1j*p_prev.imag_part_weight) -
                            (p_curr.real_part_weight + 1j*p_curr.imag_part_weight))**2
            # Combine with a scaling factor
            dist = np.sqrt(freq_dist + 0.1 * weight_dist)  # 0.1 is a tunable coefficient
            # dist = np.sqrt(
            #     (p_prev.real_part_freq - p_curr.real_part_freq) ** 2 +
            #     (p_prev.imag_part_freq - p_curr.imag_part_freq) ** 2
            # )
            # dist *= abs(p_curr.weight)  # apply weighting
            dist_matrix[i, j] = dist
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # Reorder current poles according to assignment
    new_order = [None] * n_poles
    for r, c in zip(row_ind, col_ind):
        new_order[r] = curr_poles[c]

    # --------------------------------------------------
    # 2. NEW: enforce conjugate-pair sign consistency
    # --------------------------------------------------
    # pairs: (0,1), (2,3), ...
    # for k in range(0, n_poles - 1, 2):

    #     prev_imag = prev_poles[k].imag_part_weight
    #     curr_imag = new_order[k].imag_part_weight

    #     # If the pair flipped branch, flip BOTH weights
    #     if prev_imag * curr_imag < 0:
    #         for idx in (k, k + 1):
    #             # new_order[idx].real_part_weight *= -1
    #             new_order[idx].imag_part_weight *= -1
    
    return new_order

def enforce_conjugate_weights(
        poles: List[pole_dataclass.Pole], 
        tol_freq=1e-12, tol_imag=1e-12
) -> List[pole_dataclass.Pole]:
    """
    Post-process a list of poles to ensure complex-conjugate pairs have conjugate weights.
    Purely real poles are left untouched.

    Parameters
    ----------
    poles : List[Pole]
        List of Pole objects for a single x-coordinate.
    tol_freq : float
        Tolerance for matching real parts when identifying conjugate pairs.
    tol_imag : float
        Minimum imaginary part to consider a pole as complex.

    Returns
    -------
    poles_processed : List[Pole]
        New list of Pole objects with conjugate weights enforced.
    """
    poles_processed = poles.copy()
    n_poles = len(poles_processed)
    used = set()  # Track which poles are already paired

    for i in range(n_poles):
        if i in used:
            continue

        pi = poles_processed[i]
        if abs(pi.imag_part_freq) < tol_imag:
            # Purely real pole, nothing to do
            continue

        # Find conjugate partner
        for j in range(i+1, n_poles):
            if j in used:
                continue
            pj = poles_processed[j]
            # Check if real parts match and imaginary parts are opposite
            if abs(pi.real_part_freq - pj.real_part_freq) < tol_freq and \
               abs(pi.imag_part_freq + pj.imag_part_freq) < tol_imag:
                # Enforce conjugate weights
                w1 = pi.real_part_weight + 1j*pi.imag_part_weight
                w2 = pj.real_part_weight + 1j*pj.imag_part_weight
                w_avg = (w1 + np.conj(w2)) / 2
                # Assign conjugate weights
                pi.real_part_weight = np.real(w_avg)
                pi.imag_part_weight = np.imag(w_avg)
                pj.real_part_weight = np.real(np.conj(w_avg))
                pj.imag_part_weight = np.imag(np.conj(w_avg))
                used.add(i)
                used.add(j)
                break

    return poles_processed
