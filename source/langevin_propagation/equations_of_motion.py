# source/langevin_propagation/equations_of_motion.py

from source.utils.enums import PotentialType,UnitsType
from numba import njit
import numpy as np

def build_equations_of_motion(cfg):

    model = PotentialType(cfg["potential"]["type"].lower())

    if model is PotentialType.HARMONIC:
        
        return build_harmonic_eom(cfg)
    else:

        raise ValueError(f"Other potentials not implemented yet")
    
def build_harmonic_eom(cfg):

    units_type = UnitsType(cfg["units_type"].lower())
    if units_type is UnitsType.DIMENSIONLESS:
        vib_freq = cfg["vib_freq"]
        @njit
        def x_dot(p):
            return p * vib_freq

        @njit
        def p_dot_nuc_pot(x):
            return -vib_freq * x

        @njit
        def p_dot_determ_aux(aux_fric):
            return -aux_fric * vib_freq
    elif units_type is UnitsType.DIMENSIONFULL:
        vib_freq = cfg["vib_freq"]
        mass = cfg["mass"]
        @njit
        def x_dot(p):
            return p / mass

        @njit
        def p_dot_nuc_pot(x):
            return -mass * vib_freq**2 * x

        @njit
        def p_dot_determ_aux(aux_fric):
            return -aux_fric / mass

    return {
        "x_dot": x_dot,
        "p_dot_nuc_pot": p_dot_nuc_pot,
        "p_dot_determ_aux": p_dot_determ_aux
    }