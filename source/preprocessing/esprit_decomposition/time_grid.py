# source/preprocessing/time_grid.npy
"""
For a given time vector and associated function, generates a uniform version 
of the time vector with uniform spacing and evaluates the function on this 
uniform time vector

Provides:
---------
- uniform_time_parameter_generator(time_vec)
    Given a time vector with uneven spacing, finds the minimum spacing and generates
    a uniformly spaced version of the time vector with the same minimum,maximum, but 
    the minimum spacing dt

- function_integrand_w_uniform_time(time_vec,function_integrand):
    Via interpolation, generates a version of function_integrand evaluated with a 
    uniform time_grid

Usage:
------
from source.preprocessing import decomposition

dt,n_timesteps,uniform_time_vec = decomposition.uniform_time_parameter_generator(time_vec)
uniform_function_integrand,dt,n_timesteps = decomposition.function_integrand_w_uniform_time(time_vec,function_integrand)

Notes:
------
-
"""

import numpy as np
from scipy.interpolate import CubicSpline

def uniform_time_parameter_generator(time_vec,max_decomposition_time):
    """
    Given a time vector of nonuniform steps, calculates the minimum timestep dt
    and constructs a uniform time vector from 0 to the max_decomposition_time 
    with steps of dt

    INPUTS
    -----
    time_vec : float
        vector of times with nonuniform steps

    max_decomposition_time : float
        maximum time that the uniform version goes to

    OUTPUTS
    ------
    dt : float
        minimum timestep in time_vec

    n_timesteps : integer
        number of timesteps in uniform time vector

    uniform_time_vec : float
        uniform time vector from 0 to max_decomposition_time with dt
    """
    dt = np.min(np.diff(time_vec))
    min_time = np.min(time_vec)
    max_time = np.min((np.max(time_vec),max_decomposition_time))
    n_timesteps = int((max_time - min_time)/dt)+1
    uniform_time_vec = np.linspace(min_time,max_time,n_timesteps)

    return dt,uniform_time_vec

def function_integrand_w_uniform_time(time_vec,function_integrand,max_decomposition_time):
    """
    Given a time vector of nonuniform steps and an associated vector of a function evaluated
    at those timesteps, generates a uniform version of the time and function vector evaluated
    with a uniform timestep.
    """
    function_integrand_interpolated = CubicSpline(time_vec,function_integrand)
    dt,uniform_time_vec = uniform_time_parameter_generator(time_vec,max_decomposition_time)
    uniform_function_integrand = function_integrand_interpolated(uniform_time_vec)

    return uniform_function_integrand,dt

if __name__ == "__main__":

    print()