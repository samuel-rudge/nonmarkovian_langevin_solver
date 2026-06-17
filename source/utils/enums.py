from enum import Enum

class FunctionType(Enum):
    
    FRICTION = ("friction_integrand_heom.dat", "friction", "Friction")
    CORRFUNC = ("corrfunc_integrand_heom.dat", "corrfunc", "Corrfunc")

    def __init__(self, filename, tag, plot_tag):
        
        self.filename = filename    # for input loading 
        self.tag = tag              # suffix for output saving
        self.plot_tag = plot_tag

class ForcesType(Enum):
    
    AD_FORCE = ("adiabatic_force.dat", "Adiabatic_Force")
    FRICTION = ("friction.dat", "Friction")
    CORRFUNC = ("corrfunc.dat", "Corrfunc")

    def __init__(self, markovian_filename, tag):
        
        self.markovian_filename = markovian_filename    # for input loading of markovian forces
        self.tag = tag                                  # suffix for output saving

class ElObsType(Enum):
    
    AD_CURRENT = ("current_ad.dat", "Current_Ad")
    # NA_CURRENT = ("current_na.dat", "Current_Na")

    def __init__(self, elobs_filename, tag):
        
        self.elobs_filename = elobs_filename    # for input loading of markovian forces
        self.tag = tag                                  # suffix for output saving

class PoleType(Enum):
    
    FREQ = ("freq")
    WEIGHT = ("weight")

    def __init__(self, tag):
        
        self.tag = tag              # suffix for output saving

class RealImag(Enum):
    
    REAL = ("real")
    IMAG = ("imag")

    def __init__(self, tag):
        
        self.tag = tag              # suffix for output saving    

class MagPhase(Enum):
    
    MAG = ("magnitude")
    PHASE = ("phase")

    def __init__(self, tag):
        
        self.tag = tag              # suffix for output saving    

class PhysInitConds(Enum):
    
    FIXED = ("fixed")
    WIGNER = ("wigner")
    THERMAL = ("thermal")

    def __init__(self, tag):
        
        self.tag = tag              # suffix for output saving    

class AuxInitConds(Enum):

    TRANSIENT = ("transient")
    STATIONARY = ("stationary")

    def __init__(self, tag):
    
        self.tag = tag              # suffix for output saving    

class TrajType(Enum):

    TRANSIENT = ("transient")
    STATIONARY = ("stationary")

    def __init__(self, tag):
        
        self.tag = tag              # suffix for output saving    

class PotentialType(Enum):

    HARMONIC = ("harmonic")
    MORSE = ("morse")

    def __init__(self, tag):
        
        self.tag = tag              # suffix for output saving    

class UnitsType(Enum):

    DIMENSIONLESS = ("dimensionless")
    DIMENSIONFULL = ("dimensionfull")

    def __init__(self, tag):
        
        self.tag = tag              # suffix for output saving    

class LorPoleType(Enum):

    WEIGHT = ("weight")
    FREQUENCY = ("frequency")
    WIDTH = ("width")

    def __init__(self, tag):

        self.tag = tag



