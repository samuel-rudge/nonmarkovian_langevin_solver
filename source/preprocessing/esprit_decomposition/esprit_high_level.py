# source/preprocessing/decomposition.npy
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
from source.preprocessing import esprit_decomposition
poles = decomposition.esprit_decomposition(time_vec,function_integrand)

Notes:
------
- esprit_decomposition requires a function integrand evaluated on a uniform time grid
"""

import numpy as np
import scipy
import scipy.sparse.linalg
import numpy as np
import scipy.linalg 
from scipy.optimize import minimize
from source.preprocessing import pole_dataclass
from source.preprocessing.esprit_decomposition import (
    time_grid,
    esprit_reconstruction,
    esprit_optimization,
    esprit_pole_pair
)
from typing import (
    Dict, 
    List, 
    Tuple,
    Optional
)
from numpy.typing import NDArray
from source.utils.enums import FunctionType
from source.utils import data_loader

def generate_decomposition(cfg,function_type,xdir,initial_guess):

    L_param = cfg["decomposition"]["L_param"]
    n_terms = cfg["decomposition"]["n_terms"]
    tol_esprit = float(cfg["decomposition"]["tol_esprit"])
    max_decomposition_time = cfg["decomposition"]["max_decomposition_time"]
    time_vec,function_integrand,x_coordinate = data_loader.load_function_to_decompose(function_type,xdir)
    dt,uniform_time_vec = time_grid.uniform_time_parameter_generator(time_vec,max_decomposition_time)
    uniform_function_integrand,dt = time_grid.function_integrand_w_uniform_time(time_vec,function_integrand,
                                                                                max_decomposition_time)
    initial_guess = None
    if function_type is FunctionType.FRICTION:
        uniform_function_integrand = -uniform_function_integrand
        poles = esprit_decomposition(
            uniform_function_integrand,n_terms,L_param,dt
        )
    elif function_type is FunctionType.CORRFUNC:
        poles = esprit_decomposition(
                uniform_function_integrand,n_terms,L_param,dt
            )

    return poles,uniform_time_vec

def esprit_decomposition(
        function_integrand_exact : NDArray[np.floating],
        n_terms : int,
        L_param : int,
        dt : float
) -> List[pole_dataclass.Pole]:
    """
    Perform ESPRIT pole decomposition on given function and with requested number of poles

    Parameters
    ----------
    function_integrand_exact: NDArray[np.floating]
        Exact (uniformly spaced) function to decompose
    n_terms: int
        Number of poles to use in decomposition
    L_param: int
        Size of subset to choose from
    dt: float
        Timestep
    
    Returns
    -------
    sorted_poles: Pole
        List of poles (formatted by pole_dataclass.Pole), including weight and frequency and 
        ordered by real part of frequency
    """
    
    n_timesteps = len(function_integrand_exact)
    # Construct Hankel matrix
    hankel_matrix = scipy.linalg.hankel(c=function_integrand_exact[:-(L_param+1)],
                                        r=function_integrand_exact[-(L_param+1):])
    # Decompose Hankel matrix
    U_matrix,D_matrix,V_matrix = scipy.linalg.svd(hankel_matrix,lapack_driver='gesdd')
    Y_matrix_1 = V_matrix[:n_terms,:L_param]
    Y_matrix_2 = V_matrix[:n_terms,1:(L_param+1)]
    Y_matrix_11 = Y_matrix_1.transpose()
    Y_matrix_21 = Y_matrix_2.transpose()

    A_matrix = np.matmul(np.linalg.pinv(Y_matrix_11),Y_matrix_21)
    # Solve polynomial equation for roots: z_k
    pol = np.asarray(scipy.linalg.eigvals(A_matrix))
    S_arr = scipy.sparse.linalg.svds(A_matrix,k=1,which='SM')
    j_arr = np.where(np.abs(pol)<=1e-10)[0]
    pol[j_arr] = []
    n_terms = n_terms - j_arr.size
    # Calculate frequencies gamma_k from principal value of z_k
    frequencies = -np.emath.log(pol)/dt

    # Construct Vandermonde matrix with pol
    vander_matrix = np.zeros((n_timesteps,n_terms),dtype=complex)
    for itrk in range(len(pol)):
        vander_matrix[:,itrk] = np.transpose(pol[itrk]**(np.arange(0,n_timesteps)))

    # Calculate weights eta_k from least squares solution of Vandermonde matrix
    weights = np.linalg.lstsq(vander_matrix,function_integrand_exact,rcond=None)[0]

    poles = pole_dataclass.create_poles_from_arrays(frequencies,weights)

    return poles

def esprit_smoothing(
    cfg: Dict,
    function_type: FunctionType,
    xdir,
    input_poles: List[pole_dataclass.Pole],
    prev_poles: List[pole_dataclass.Pole]
):


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

    L_param = cfg["decomposition"]["L_param"]
    n_terms = cfg["decomposition"]["n_terms"]
    tol_esprit = float(cfg["decomposition"]["tol_esprit"])
    max_decomposition_time = cfg["decomposition"]["max_decomposition_time"]
    time_vec,function_integrand,x_coordinate = data_loader.load_function_to_decompose(function_type,xdir)
    dt,uniform_time_vec = time_grid.uniform_time_parameter_generator(time_vec,max_decomposition_time)
    uniform_function_integrand,dt = time_grid.function_integrand_w_uniform_time(time_vec,function_integrand,
                                                                                max_decomposition_time)
    
    weights,freq = esprit_optimization.generate_optimized_parameters(
        time_vec=uniform_time_vec,
        target=uniform_function_integrand, 
        initial_guess=input_poles,
        prev_poles=prev_poles
    )
    
    poles = pole_dataclass.create_poles_from_arrays(freq,weights)

    return poles

def nonlinear_refinement_w_initial_guess(
    function_integrand_exact: NDArray[np.floating],
    dt: float,
    initial_guess: List[pole_dataclass.Pole],
    maxiter: int = 2000
) -> List["pole_dataclass.Pole"]:
    """
    Nonlinear refinement of ESPRIT poles using scalar objective minimization.

    No bounds, no positivity enforcement.
    Designed for real-valued signals with conjugate poles.
    """

    t = np.arange(len(function_integrand_exact)) * dt
    y = function_integrand_exact
    n = len(initial_guess)

    # ---------------------------------------
    # Parameter vector: [Re(a), Im(a), Re(b), Im(b)]
    # ---------------------------------------
    p0 = np.zeros(4 * n)
    for k, pole in enumerate(initial_guess):
        p0[4*k + 0] = pole.weight.real
        p0[4*k + 1] = pole.weight.imag
        p0[4*k + 2] = pole.frequency.real
        p0[4*k + 3] = pole.frequency.imag

    # ---------------------------------------
    # Scalar objective
    # ---------------------------------------
    def objective(p):
        model = np.zeros_like(y, dtype=complex)

        for k in range(n):
            a = p[4*k + 0] + 1j * p[4*k + 1]
            b = p[4*k + 2] + 1j * p[4*k + 3]

            # Safe exponential
            exp_term = np.exp(np.clip(-b * t, -700, 700))
            model += a * exp_term

        model = np.real(model)
        err = y - model
        return np.mean(err**2)

    # ---------------------------------------
    # Optimize
    # ---------------------------------------
    res = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "ftol": 1e-9
        }
    )

    p_opt = res.x

    # ---------------------------------------
    # Rebuild poles (including conjugates)
    # ---------------------------------------
        
    weights = [p_opt[4*k + 0] + 1j * p_opt[4*k + 1] for k in range(n)]
    frequencies = [p_opt[4*k + 2] + 1j * p_opt[4*k + 3] for k in range(n)]
    refined_poles = pole_dataclass.create_poles_from_arrays(frequencies,weights)
    
    return refined_poles


# def nonlinear_refinement_w_initial_guess(
#     function_integrand_exact: NDArray[np.floating],
#     dt: float,
#     initial_guess: List["pole_dataclass.Pole"],
#     lam: float = 0.0,
#     max_nfev: int = 50
# ) -> List["pole_dataclass.Pole"]:
#     """
#     Nonlinear least-squares refinement for a real signal, enforcing conjugate symmetry
#     of complex poles and weights.

#     Parameters
#     ----------
#     function_integrand_exact : array of float
#         Real-valued signal f(t)
#     dt : float
#         Time step
#     initial_guess : list of Pole
#         Poles from previous x (frequency and weight)
#         Must include only *one* of each conjugate pair
#     lam : float, optional
#         Smoothness regularization strength
#     max_nfev : int, optional
#         Maximum optimizer iterations

#     Returns
#     -------
#     poles : list of Pole
#         Refined poles (all poles including conjugates)
#     """

#     n_unique = len(initial_guess)  # number of unique poles (half of conjugate pairs)
#     n_timesteps = len(function_integrand_exact)
#     t = dt * np.arange(n_timesteps)

#     # ------------------------------------------------------------
#     # Pack initial guess: theta = [alpha, omega, |a|, phase(a)]
#     # ------------------------------------------------------------
#     theta0 = np.zeros(4 * n_unique)
#     for j, pole in enumerate(initial_guess):
#         b = pole.frequency
#         a = pole.weight
#         theta0[j] = b.real                 # alpha
#         theta0[j + n_unique] = b.imag      # omega
#         theta0[j + 2 * n_unique] = np.abs(a)       # amplitude magnitude
#         theta0[j + 3 * n_unique] = np.angle(a)     # amplitude phase

#     theta_prev = theta0.copy()

#     # ------------------------------------------------------------
#     # Residual function
#     # ------------------------------------------------------------
#     def residual(theta):
#         alpha = theta[0:n_unique]
#         omega = theta[n_unique:2 * n_unique]
#         amp = theta[2*n_unique:3*n_unique]
#         phase = theta[3*n_unique:4*n_unique]

#         model = np.zeros_like(function_integrand_exact, dtype=float)

#         for j in range(n_unique):
#             b = alpha[j] + 1j * omega[j]
#             a = amp[j] * np.exp(1j * phase[j])
#             exp_arg = -b * t
#             # clip to safe range
#             exp_arg = np.clip(exp_arg, -700, 700)  # np.exp(700) ~ 1e304
#             # Add both the pole and its conjugate contribution
#             model += 2 * np.real(a * np.exp(exp_arg))

#         r = model - function_integrand_exact

#         # Optional smoothness regularization (only poles)
#         if lam > 0.0:
#             res_reg = np.sqrt(lam) * (theta[:2*n_unique] - theta_prev[:2*n_unique])
#             r = np.concatenate([r, res_reg])

#         return r

#     # ------------------------------------------------------------
#     # Nonlinear least squares
#     # ------------------------------------------------------------
#     result = scipy.optimize.least_squares(
#         residual,
#         theta0,
#         method="dogbox",
#         max_nfev=max_nfev,
#         ftol=1e-6
#     )

#     theta_opt = result.x

#     # ------------------------------------------------------------
#     # Unpack optimized parameters
#     # ------------------------------------------------------------
#     alpha = theta_opt[0:n_unique]
#     omega = theta_opt[n_unique:2*n_unique]
#     amp = theta_opt[2*n_unique:3*n_unique]
#     phase = theta_opt[3*n_unique:4*n_unique]

#     poles_freqs = []
#     poles_weights = []

#     for j in range(n_unique):
#         b = alpha[j] + 1j * omega[j]
#         a = amp[j] * np.exp(1j * phase[j])
#         # store both the pole and its conjugate
#         poles_freqs.append(b)
#         poles_weights.append(a)
#         poles_freqs.append(np.conj(b))
#         poles_weights.append(np.conj(a))

#     poles = pole_dataclass.create_poles_from_arrays(
#         np.array(poles_freqs),
#         np.array(poles_weights)
#     )

#     return poles
