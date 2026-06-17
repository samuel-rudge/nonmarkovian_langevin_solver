# source/langevin_propagation/prepare_initial_conditions.py

"""
Docstring for source.langevin_propagation.prepare_initial_conditions
"""

from source.utils.enums import (
    PhysInitConds,
    AuxInitConds
)
import numpy as np
from numba import njit

def fixed_physical_ics(n_traj,n_vib,ic_mean):

    physical_state = np.zeros((n_traj,n_vib,2),dtype=float)
    for itr_vib in range(n_vib):
        physical_state[:,itr_vib,:] = ic_mean[itr_vib,:]

    return physical_state

def wigner_physical_ics(n_traj,n_vib,ic_mean,ic_var):

    physical_state = np.zeros((n_traj,n_vib,2),dtype=float)
    for itr_vib in range(n_vib):
        physical_state[:,itr_vib,:] = draw_random_numbers_box_muller(
            ic_mean[itr_vib,:],ic_var[itr_vib,0],n_traj
            )

    return physical_state

def draw_random_numbers_box_muller(ic_mean, w_eff, n_itr):

    u1 = np.random.uniform(size = n_itr)
    u2 = np.random.uniform(size = n_itr)
    sigma_q = np.sqrt(1/(2*w_eff))
    sigma_p = np.sqrt(w_eff/2)
    mu_q = ic_mean[0]
    mu_p = ic_mean[1]
    r_q = sigma_q*np.sqrt(-np.log(u1))
    r_p = sigma_p*np.sqrt(-np.log(u1))
    theta = 2 * np.pi * u2
    q_draw = r_q * np.cos(theta)+mu_q
    p_draw = r_p * np.sin(theta)+mu_p

    return np.column_stack((q_draw,p_draw))

def thermal_physical_ics(n_traj,n_vib,ic_mean,ic_var,temp):

    raise NotImplementedError(
        "Thermal ICs are not available yet."
    )

def prepare_physical_ics(cfg):

    init_cond_cfg = cfg["simulation"]
    ic_type_str = init_cond_cfg["phys_ic_type"]
    try:
        ic_type = PhysInitConds(ic_type_str.lower())
    except ValueError:
        raise ValueError(f"Unknown ic_type: {ic_type_str}")
    
    n_traj = init_cond_cfg["n_traj"]
    n_vib = cfg["n_vib"]
    ic_mean = np.atleast_2d(init_cond_cfg["ic_mean"])
    if ic_type is PhysInitConds.FIXED:
        physical_state = fixed_physical_ics(n_traj,n_vib,ic_mean)
    elif ic_type is PhysInitConds.WIGNER:
        ic_var = np.atleast_2d(init_cond_cfg["ic_var"])
        physical_state = wigner_physical_ics(n_traj,n_vib,ic_mean,ic_var)
    elif ic_type is PhysInitConds.THERMAL:
        ic_var = init_cond_cfg["ic_var"]
        init_vib_temp = init_cond_cfg["init_vib_temp"]
        physical_state = thermal_physical_ics(n_traj,n_vib,ic_mean,ic_var,init_vib_temp)

    return physical_state

# @njit
def generate_determ_matrices(interp_friction_arr,itr_pole,x):

    freq_real = interp_friction_arr[0][0][itr_pole](x)
    freq_imag = interp_friction_arr[0][1][itr_pole](x)
    weights_real = interp_friction_arr[1][0][itr_pole](x)
    weights_imag = interp_friction_arr[1][1][itr_pole](x)
    b_arr = np.array([[freq_real,-freq_imag],[freq_imag,freq_real]],dtype=float)
    a_arr = np.array([weights_real,weights_imag],dtype=float)

    return a_arr,b_arr

def prepare_determ_aux_ics(
        cfg,
        physical_state,
        interp_forces_arr
):

    init_cond_cfg = cfg["simulation"]
    ic_type_str = init_cond_cfg["aux_ic_type"]
    try:
        ic_type = AuxInitConds(ic_type_str.lower())
    except ValueError:
        raise ValueError(f"Unknown ic_type: {ic_type_str}")
    
    n_traj = init_cond_cfg["n_traj"]
    n_vib = cfg["n_vib"]
    n_terms = cfg["decomposition"]["n_terms"]
    determ_aux_state = np.zeros((n_traj,n_terms,n_vib,2),dtype=float)
    if ic_type is AuxInitConds.STATIONARY:
        for itr_vib in range(n_vib):
            physical_state_vib = physical_state[:,itr_vib,:]
            for itr_pole in range(n_terms):
                for itr_traj in range(n_traj):
                    a_arr,b_arr = generate_determ_matrices(
                        interp_forces_arr[1],
                        itr_pole,
                        physical_state_vib[itr_traj,0]
                        )
                    determ_aux_state[itr_traj,itr_pole,itr_vib,:] = \
                        np.linalg.solve(b_arr,a_arr)*physical_state_vib[itr_traj,1]

    return determ_aux_state

def generate_sigma(interp_corrfunc_arr,itr_pole,x):

    # freq_real = interp_corrfunc_arr[0][0][itr_pole](x)
    weights_real = interp_corrfunc_arr[1][0][itr_pole](x)
    # weights_imag = interp_corrfunc_arr[1][1][itr_pole](x)
    # weights_mag = np.sqrt(weights_real**2 + weights_imag**2)
    # stat_sigma = np.sqrt(2 * freq_real * weights_mag)
    stat_sigma = np.sqrt(weights_real)

    return stat_sigma

def prepare_stoch_aux_ics(
        cfg,
        physical_state,
        interp_forces_arr
):

    init_cond_cfg = cfg["simulation"]
    ic_type_str = init_cond_cfg["aux_ic_type"]
    try:
        ic_type = AuxInitConds(ic_type_str.lower())
    except ValueError:
        raise ValueError(f"Unknown ic_type: {ic_type_str}")
    
    base_seed = init_cond_cfg["base_seed"]
    # rng = np.random.default_rng(base_seed)
    n_traj = init_cond_cfg["n_traj"]
    n_vib = cfg["n_vib"]
    n_terms = cfg["decomposition"]["max_lorentzian_terms"]
    stoch_aux_state = np.zeros((n_traj,n_terms,n_vib,2),dtype=float)
    traj_seeds = []
    if ic_type is AuxInitConds.STATIONARY:
        for itr_vib in range(n_vib):
            physical_state_vib = physical_state[:,itr_vib,:]
            for itr_traj in range(n_traj):
                # per-trajectory RNG
                traj_seed = base_seed + 1000003 * itr_traj
                traj_seeds.append(traj_seed)
                traj_rng = np.random.default_rng(traj_seed)
                for itr_pole in range(n_terms):
                    stat_sigma = generate_sigma(interp_forces_arr[2], itr_pole, physical_state_vib[itr_traj,0])
                    stoch_aux_state[itr_traj, itr_pole, itr_vib, :] = \
                        traj_rng.normal(
                            0.0,
                            stat_sigma,
                            size=2
                        )
        
    return stoch_aux_state,traj_seeds

def generate_all_ics(cfg,interp_forces_arr):
    
    phys_ic = prepare_physical_ics(cfg)
    markovian_value = cfg["simulation"]["markovian"]
    if not markovian_value:
        determ_aux_ic = prepare_determ_aux_ics(cfg,phys_ic,interp_forces_arr)
        stoch_aux_ic,traj_seeds = prepare_stoch_aux_ics(cfg,phys_ic,interp_forces_arr)
        n_traj = phys_ic.shape[0]
        initial_conditions_arr = [{
            "physical_state": phys_ic[i],
            "determ_aux_state": determ_aux_ic[i],
            "stoch_aux_state": stoch_aux_ic[i] 
        } for i in range(n_traj)]
    else: 
        n_traj = phys_ic.shape[0]
        initial_conditions_arr = [{
            "physical_state": phys_ic[i]
        } for i in range(n_traj)]
    
    return initial_conditions_arr,traj_seeds

if __name__ == "__main__":

    import numpy as np
    from source.utils.config import load_config
    from source.langevin_propagation.prepare_forces import load_and_interpolate_forces
    from matplotlib import pyplot as plt
    
    cfg = load_config()
    physical_state = prepare_physical_ics(cfg)
    interp_forces_arr,grid_limits = load_and_interpolate_forces(cfg,0.0)
    determ_aux_state = prepare_determ_aux_ics(cfg,physical_state,interp_forces_arr)
    stoch_aux_state = prepare_stoch_aux_ics(cfg,physical_state,interp_forces_arr)
    # print(stoch_aux_state[:,0,0,:])
    # n, bins, patches = plt.hist(physical_state[:,0],bins = 100)
    # n, bins, patches = plt.hist(physical_state[:,1],bins = 100)
    # n, bins, patches = plt.hist(stoch_aux_state[:,0,0,0][0],bins = 100)
    # n, bins, patches = plt.hist(stoch_aux_state[:,1,0,0][0],bins = 100)
    # n, bins, patches = plt.hist(stoch_aux_state[:,2,0,0][0],bins = 100)
    # n, bins, patches = plt.hist(stoch_aux_state[:,5,0,0][0],bins = 100)
    # plt.show()
