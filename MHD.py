import numpy as np 
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import TypedDict
from enum import Enum

PARAM_DICT_TYPING =  TypedDict("Params", {"gamma": np.float64, "dx": np.float64, "Courant": 0.5})
#  Params = {"gamma": 2.0, "dx": 0.05, "Courant":0.5}

# NB I'm assuming that the dim(U) is Nx3

class PrimativeIndex(Enum):
    DENSITY = 0
    VELOCITY = 1
    PRESSURE = 2

class ConservativeIndex(Enum):
    DENSITY = 0
    MOMENTUM_DENSITY = 1
    ENERGY = 2


def internal_energy_conservative(U: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    rho = U[:ConservativeIndex.DENSITY.value]
    assert(np.any(rho==0))
    v = U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/rho
    E = U[:,ConservativeIndex.ENERGY.value]
    # Total energy
    #   E = \rho *e + 0.5*\rho * v**2
    # Inverting 
    # (E-0.5 \rho v^2)/\rho = e
    return (E-0.5*rho*np.power(v,2))/rho    

def internal_energy_primitive(W: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # primitive variables 
    # W = (\rho, v, P)
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # Invert this: e = P/((\gamma-1) * \rho)
    return W[:,PrimativeIndex.PRESSURE.value]/( (global_parameters["gamma"]-1) * W[:,PrimativeIndex.DENSITY.value])

def equation_of_state_conservative(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # Return pressure given the conservative variables 
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # e is the internal energy
    # gamma is adiabatic index 
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    return (global_parameters["gamma"]-1)*U[:,ConservativeIndex.DENSITY.value]*internal_energy_conservative(U)

def conservative_to_primitive(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = U[:ConservativeIndex.DENSITY.value]
    pressure = equation_of_state_conservative(U, global_parameters)
    assert(np.any(U[:,ConservativeIndex.DENSITY.value]==0))
    return np.stack([rho, U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/ rho, pressure], axis=1) 

def primitive_to_conservative(W: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = W[:,PrimativeIndex.DENSITY.value]
    velocity = W[:,PrimativeIndex.VELOCITY.value]
    # Total energy
    #   E = \rho *e + 0.5*\rho * v**2
    E = rho * internal_energy_primitive(W, global_parameters) + 0.5*rho*np.power(velocity,2)
    return np.stack([rho, rho*velocity, E], axis= 1) 

def flux_from_conservative(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    primitive = conservative_to_primitive(U,global_parameters)
    F_0 = U[:,ConservativeIndex.MOMENTUM_DENSITY.value]
    F_1 = F_0*primitive[:,PrimativeIndex.VELOCITY.value]+primitive[:,PrimativeIndex.PRESSURE.value]
    F_2 = (U[:,ConservativeIndex.ENERGY.value]+primitive[:,PrimativeIndex.PRESSURE.value])*primitive[:PrimativeIndex.VELOCITY.value]
    return np.stack([F_0,F_1,F_2], axis=1)

def alpha_plus_minus(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    primitives = conservative_to_primitive(U, global_parameters)
    c_s = np.sqrt(global_parameters["gamma"]*primitives[:, PrimativeIndex.PRESSURE.value]/ primitives[:, PrimativeIndex.DENSITY.value])
    lambda_plus = primitives[:,PrimativeIndex.VELOCITY.value]+c_s
    lambda_minus = primitives[:,PrimativeIndex.VELOCITY.value]-c_s
    # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for both + and - 
    zeros = np.zeros(lambda_plus.shape[0])
    # First, the plus case   
    # Augment the eigenvalue array with 0 at the start and end
    lambda_plus_right = np.insert(lambda_plus, lambda_plus.shape[0], 0)
    lambda_plus_padded = np.insert(lambda_plus_right, 0, 0) 
    # Grab the left and right speeds from the padded array
    alpha_plus_candidates = np.stack(zeros, lambda_plus_padded[:-1], lambda_plus_padded[0:], axis=1)    
    # Calculate max across each row
    alpha_plus = np.max(alpha_plus_candidates, axis=1)
    # minus case
    lambda_minus_right = np.insert(lambda_minus, lambda_minus.shape[0], 0)
    lambda_minus_padded = np.insert(lambda_minus_right, 0, 0) 
    alpha_minus_candidates = np.stack(zeros, lambda_minus_padded[:-1], lambda_minus_padded[0:], axis=1)    
    alpha_minus = np.max(alpha_minus_candidates, axis=1)
    return (alpha_minus, alpha_plus)

def spatial_derivative(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    cell_flux = flux_from_conservative(U, global_parameters)
    alpha_minus, alpha_plus = alpha_plus_minus(U, global_parameters)
    alpha_sum = alpha_minus+alpha_plus 
    assert(np.any(alpha_sum != 0))
    alpha_prod = alpha_minus*alpha_plus
    # Augment the array to have 0 flux at the boundaries (ie. copy the edge values to the ghost cells)
    U_right_pad = np.insert(U, U.shape[0], U[-1])
    U_padded  = np.insert(U_right_pad, 0, U_right_pad[0]) 
    # Augment flux array as well 
    cell_flux_right_pad = np.insert(cell_flux, cell_flux.shape[0], cell_flux[-1])
    cell_flux_padded  = np.insert(cell_flux_right_pad, 0, cell_flux_right_pad[0]) 
    cell_flux_plus_half = (alpha_plus*cell_flux_padded[1:-1]+ alpha_minus*cell_flux_padded[2:]-alpha_prod*(U_padded[2:]-U_padded[1:-1]))/alpha_prod 
    cell_flux_minus_half = (alpha_plus*cell_flux_padded[:,-2]+alpha_minus*cell_flux_padded[1:-1]-alpha_prod*(U_padded[1:-1]-U_padded[2:]))/alpha_prod
    return -(cell_flux_plus_half-cell_flux_minus_half)/global_parameters["dx"], alpha_plus, alpha_minus

def calc_dt(alpha_plus: npt.ArrayLike, alpha_minus:npt.ArrayLike, global_parameters: PARAM_DICT_TYPING):
    max_alpha = np.max( [alpha_plus, alpha_minus]) 
    return global_parameters["Courant"]*global_parameters["dx"]/max_alpha

def update(U: npt.ArrayLike, current_time: np.float64, global_parameters: PARAM_DICT_TYPING) -> tuple[np.float64, npt.NDArray]:
    flux_change, alpha_plus, alpha_minus = spatial_derivative(U, global_parameters)
    dt = calc_dt(alpha_plus, alpha_minus, global_parameters)
    return current_time+dt, U+flux_change*dt

def SodShockInitialization(rho_l: np.float64, v_l: np.float64, P_l: np.float64,
                           rho_r:np.float64, v_r: np.float64, P_r:np.float64,
                           N: np.int64 = 100) -> tuple[npt.NDArray, PARAM_DICT_TYPING]:
    Params = {"gamma": 1.4, "dx": 1/N, "Courant":0.5}
    primitives = np.zeros( (N,3)) 
    grid = np.linspace(0.0,1.0,N)
    lower_half = grid<0.5 
    upper_half = grid>=0.5
    primitives[lower_half, PrimativeIndex.DENSITY.value] = rho_l
    primitives[lower_half, PrimativeIndex.VELOCITY.value] = v_l
    primitives[lower_half, PrimativeIndex.PRESSURE.value] = P_l
    primitives[upper_half, PrimativeIndex.DENSITY.value] = rho_r
    primitives[upper_half, PrimativeIndex.VELOCITY.value] = v_r
    primitives[upper_half, PrimativeIndex.PRESSURE.value] = P_r
    return primitive_to_conservative(primitives, Params), Params
    
if __name__ == "__main__":
    initial_cond, param_dict = SodShockInitialization(1.0,0.0,1.0, 0.125, 0.0, 0.1)
    print(initial_cond.shape, param_dict)
