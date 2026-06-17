# source/langevin_propagation/aboba_implementation.py

import numpy as np
from typing import (
    Dict, 
    List, 
    Tuple,
    Callable
)
from numpy.typing import NDArray #type: ignore
from numba import njit
import logging

def compute_determ_ab_term(
        physical_state: NDArray[np.floating],
        dt: np.floating,
        adiabatic_force_func: Callable[[float], float],
        x_dot_func,
        p_dot_nuc_pot_func,
        grid_limits: NDArray[np.floating]
):
    """
    Compute A and B components of the ABOBA algorithm
    """
    
    physical_state[0,0] += dt*x_dot_func(physical_state[0,1])
    check_grid_limits(physical_state[0,0],grid_limits)
    physical_state[0,1] += dt*(adiabatic_force_func(physical_state[0,0]) + 
                                p_dot_nuc_pot_func(physical_state[0,0]))
    
    return physical_state

def compute_determ_ba_term(
        physical_state: NDArray[np.floating],
        dt: np.floating,
        adiabatic_force_func: Callable[[float], float],
        x_dot_func,
        p_dot_nuc_pot_func,
        grid_limits: NDArray[np.floating]
):
    """
    Compute A and B components of the ABOBA algorithm
    """
    
    physical_state[0,1] += dt*(adiabatic_force_func(physical_state[0,0]) + 
                                p_dot_nuc_pot_func(physical_state[0,0]))
    physical_state[0,0] += dt*x_dot_func(physical_state[0,1])
    check_grid_limits(physical_state[0,0],grid_limits)

    return physical_state

def generate_friction_arr(interp_friction_arr,physical_state):

    x = physical_state[0,0]
    n_poles = len(interp_friction_arr[0][0])
    friction_arr = np.zeros((2,2,n_poles),dtype=float)
    for itr_pole in range(len(interp_friction_arr[0][0])):
        friction_arr[0][0][itr_pole] = interp_friction_arr[0][0][itr_pole](x)
        friction_arr[0][1][itr_pole] = interp_friction_arr[0][1][itr_pole](x)
        friction_arr[1][0][itr_pole] = interp_friction_arr[1][0][itr_pole](x)
        friction_arr[1][1][itr_pole] = interp_friction_arr[1][1][itr_pole](x)

    return friction_arr

def generate_corrfunc_arr(interp_corrfunc_arr,physical_state):

    x = physical_state[0,0]
    n_poles = len(interp_corrfunc_arr[0][0])
    corrfunc_arr = np.zeros((2,2,n_poles),dtype=float)
    for itr_pole in np.arange(len(interp_corrfunc_arr[0][0])):
        corrfunc_arr[0][0][itr_pole] = interp_corrfunc_arr[0][0][itr_pole](x)
        corrfunc_arr[0][1][itr_pole] = interp_corrfunc_arr[0][1][itr_pole](x)
        corrfunc_arr[1][0][itr_pole] = interp_corrfunc_arr[1][0][itr_pole](x)
        corrfunc_arr[1][1][itr_pole] = interp_corrfunc_arr[1][1][itr_pole](x)

    return corrfunc_arr

@njit
def update_determ_aux_state(
        physical_state,
        determ_aux_state,
        dt,
        friction_arr
):

    p = physical_state[0,1]
    for itr_pole,aux_fric in enumerate(determ_aux_state):
        aux_fric = aux_fric.flatten()
        freq_real = friction_arr[0][0][itr_pole]
        freq_imag = friction_arr[0][1][itr_pole]
        weight_real = friction_arr[1][0][itr_pole]
        weight_imag = friction_arr[1][1][itr_pole]

        # Define matrices
        alpha = freq_real
        beta = freq_imag
        # A = np.array([[-alpha,  beta],
        #               [-beta, -alpha]])
        a_vec = np.array([weight_real, weight_imag])

        # Matrix exponential
        expA_dt = np.exp(-alpha * dt) * np.array([[np.cos(beta*dt), np.sin(beta*dt)],
                                                 [-np.sin(beta*dt), np.cos(beta*dt)]])

        # Inverse of A
        det = alpha**2 + beta**2
        A_inv = (1.0/det) * np.array([[-alpha, -beta],
                                      [ beta, -alpha]])

        aux_fric_new = expA_dt @ aux_fric + A_inv @ (expA_dt - np.eye(2)) @ a_vec * p
        determ_aux_state[itr_pole] = aux_fric_new
    
    return determ_aux_state

def generate_physical_state_midpoint(physical_state,determ_aux_state,
                        p_dot_determ_aux_func,dt,x_dot_func):

    physical_state_midpoint = np.copy(physical_state)
    physical_state_midpoint = update_p_due_to_friction(
        physical_state,
        determ_aux_state,
        p_dot_determ_aux_func,
        dt/2)
    physical_state_midpoint[0,0] += dt/4*x_dot_func(physical_state[0,1])

    return physical_state_midpoint

def update_p_due_to_friction(
        physical_state,
        determ_aux_state,
        p_dot_determ_aux_func,
        dt
):

    for aux_fric in determ_aux_state:
        physical_state[0,1] += p_dot_determ_aux_func(aux_fric[0,0])*dt
    
    return physical_state

@njit
def update_stoch_matrices(
        stoch_matrix_inputs,
        dt,
        decay_rot_mat_exp
):
    
    decay_term = np.exp(-stoch_matrix_inputs[0]*dt)
    cos_rot = np.cos(stoch_matrix_inputs[1]*dt)
    sin_rot = np.sin(stoch_matrix_inputs[1]*dt)
    decay_rot_mat_exp[0,0] = decay_term*cos_rot
    decay_rot_mat_exp[0,1] = decay_term*sin_rot
    decay_rot_mat_exp[1,0] = -decay_term*sin_rot
    decay_rot_mat_exp[1,1] = decay_term*cos_rot
    stoch_cov = np.sqrt(stoch_matrix_inputs[2]*(1 - np.exp(-2*stoch_matrix_inputs[0]*dt)))

    return stoch_cov,decay_rot_mat_exp

@njit
def update_stoch_aux_state(
        stoch_aux_state,
        dt,
        corrfunc_arr,
        rand_arr
):
    
    decay_rot_mat_exp = np.zeros((2,2),dtype=np.float64)
    n_poles = stoch_aux_state.shape[0]
    for itr_pole in range(n_poles):
        aux_stoch = stoch_aux_state[itr_pole].flatten()
        freq_real = corrfunc_arr[0][0][itr_pole]
        freq_imag = corrfunc_arr[0][1][itr_pole]
        weights_real = corrfunc_arr[1][0][itr_pole]
        stoch_matrix_inputs = [freq_real,freq_imag,weights_real]
        stoch_cov,decay_rot_mat_exp = update_stoch_matrices(stoch_matrix_inputs,dt,decay_rot_mat_exp)
        stoch_sample = stoch_cov*rand_arr[itr_pole,:]
        aux_stoch = decay_rot_mat_exp @ aux_stoch + stoch_sample
        stoch_aux_state[itr_pole] = aux_stoch

    return stoch_aux_state

@njit
def generate_stoch_force(
        stoch_aux_state
):
    
    stoch_force = 0
    for aux_stoch in stoch_aux_state:
        stoch_force += aux_stoch[0,0]

    return stoch_force

@njit
def update_p_due_to_stoch_force(
        physical_state,
        stoch_aux_state,
        dt
):

    stoch_force = generate_stoch_force(stoch_aux_state)
    physical_state[0,1] += stoch_force*dt
    
    return physical_state,stoch_force

def check_grid_limits(x,grid_limits):
     
    if (x < grid_limits[0]):
        raise ValueError(f"x-coordinate is below grid limit {grid_limits[0]}")
    if(x > grid_limits[1]):
        raise ValueError(f"x-coordinate is above grid limit {grid_limits[1]}")

def compute_full_o_term(
        physical_state: NDArray[np.floating],
        determ_aux_state: NDArray[np.floating],
        stoch_aux_state: NDArray[np.floating],
        dt: np.floating,
        interp_func_arr,
        p_dot_determ_aux_func,
        x_dot_func,
        rng
):

    friction_arr = generate_friction_arr(interp_func_arr[1],physical_state)
    corrfunc_arr = generate_corrfunc_arr(interp_func_arr[2],physical_state)
    determ_aux_state = update_determ_aux_state(physical_state,determ_aux_state,
                        dt/2,friction_arr)
    # physical_state_midpoint = generate_physical_state_midpoint(physical_state,determ_aux_state,
    #                     p_dot_determ_aux_func,dt,x_dot_func)
    # corrfunc_arr = generate_corrfunc_arr(interp_func_arr[2],physical_state_midpoint)
    rand_arr = rng.normal(0.0,1.0,size=(len(stoch_aux_state),2))
    stoch_aux_state = update_stoch_aux_state(stoch_aux_state,dt/2,corrfunc_arr,rand_arr)
    physical_state = update_p_due_to_friction(physical_state,determ_aux_state,p_dot_determ_aux_func,dt)
    physical_state,stoch_force = update_p_due_to_stoch_force(physical_state,stoch_aux_state,dt)
    # physical_state_midpoint = generate_physical_state_midpoint(physical_state,determ_aux_state,
    #                     p_dot_determ_aux_func,dt,x_dot_func)
    # corrfunc_arr =  generate_corrfunc_arr(interp_func_arr[2],physical_state_midpoint)
    rand_arr = rng.normal(0.0,1.0,size=(len(stoch_aux_state),2))
    stoch_aux_state = update_stoch_aux_state(stoch_aux_state,dt/2,corrfunc_arr,rand_arr)
    determ_aux_state = update_determ_aux_state(physical_state,determ_aux_state,
                        dt/2,friction_arr)

    return physical_state,determ_aux_state,stoch_aux_state,stoch_force

def propagate_aboba_onestep(
        input,
        interp_func_array,
        dimfull_func_arr,
        dt,
        grid_limits,
        rng
):

    physical_state = input["physical_state"]
    determ_aux_state = input["determ_aux_state"]
    stoch_aux_state = input["stoch_aux_state"]
    physical_state = compute_determ_ab_term(
                                physical_state,dt/2,interp_func_array[0],
                                dimfull_func_arr["x_dot"],
                                dimfull_func_arr["p_dot_nuc_pot"],
                                grid_limits
                                )
    physical_state,determ_aux_state,stoch_aux_state,stoch_force = compute_full_o_term(
                                physical_state,determ_aux_state,
                                stoch_aux_state,dt,interp_func_array,
                                dimfull_func_arr["p_dot_determ_aux"],
                                dimfull_func_arr["x_dot"],
                                rng
                                )
    physical_state = compute_determ_ba_term(
                                physical_state,dt/2,interp_func_array[0],
                                dimfull_func_arr["x_dot"],
                                dimfull_func_arr["p_dot_nuc_pot"],
                                grid_limits
                                )
    
    # print(physical_state[0,:])

    state_output = {
        "physical_state": physical_state,
        "determ_aux_state": determ_aux_state,
        "stoch_aux_state": stoch_aux_state
    }
    observable_output = {
        "stoch_force": stoch_force
    }

    return state_output,observable_output

def propagate_aboba_markovian_onestep(
            physical_state,
            interp_forces_arr,
            dimfull_func_arr,
            dt,
            grid_limits,Kval):

    factor = np.sqrt(dt/Kval)
    n_random_numbers = 1
    
    physical_state = compute_determ_ab_term(physical_state,dt/2,interp_forces_arr[0],
                                  dimfull_func_arr["x_dot"],
                                  dimfull_func_arr["p_dot_nuc_pot"],
                                  grid_limits)

    friction = np.abs(interp_forces_arr[1](physical_state[0,0]))*0.03
    corrfunction = np.sqrt(2*np.abs(interp_forces_arr[2](physical_state[0,0])))
    G_term = np.exp(-dt*friction/(2*Kval))
    for itrj in range(Kval):
        physical_state[0,1] = G_term*G_term*physical_state[0,1] + factor*G_term*corrfunction*np.random.randn(n_random_numbers)
    
    physical_state = compute_determ_ba_term(physical_state,dt/2,interp_forces_arr[0],
                                  dimfull_func_arr["x_dot"],
                                  dimfull_func_arr["p_dot_nuc_pot"],
                                  grid_limits)

    return physical_state
