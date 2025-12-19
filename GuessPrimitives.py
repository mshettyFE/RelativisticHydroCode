from HydroCore import SimParams
import numpy.typing as npt
import numpy as np
from metrics.Metric import Metric, WhichCacheTensor
from GridInfo import GridInfo, WeightType
from HelperFunctions import index_conservative_var, index_primitive_var, PrimitiveIndex, ConservativeIndex
from EquationOfState import *


def root_finding_func(guess: npt.ArrayLike,
                              U_cart: npt.ArrayLike, 
                               metric: Metric,
                                sim_params: SimParams,
                                grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    flux = U_cart[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    flux_squared = metric.three_vector_mag_squared(flux, grid_info, WeightType.CENTER, sim_params)
    D = index_conservative_var(U_cart, ConservativeIndex.DENSITY, n_spatial_dim)
    Tau = index_conservative_var(U_cart, ConservativeIndex.TAU, n_spatial_dim)
    z = Tau+guess+D
    z = np.maximum(z, 1e-12)
    v_mag_2 = flux_squared/np.power(z,2)
    v_mag_2 = np.minimum(v_mag_2, 1.0 - 1e-10)
    W2 = np.power(1-v_mag_2,-1)
    W = np.sqrt(W2)
    epsilon = (Tau+D*(1-W)+guess*(1.0-W2))/(D*W)
    epsilon = np.maximum(epsilon, 1e-10)
    rho = D/W
    guess_pressure = equation_of_state_epsilon(sim_params, epsilon,rho )
    out  = guess_pressure - guess
    return out

def construct_primitives_from_guess(guess:npt.ArrayLike,
                                     U_cart: npt.ArrayLike, metric: Metric,
                                     sim_params: SimParams,
                                     grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    output = np.zeros(U_cart.shape)
    flux = U_cart[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    flux_squared = metric.three_vector_mag_squared(flux, grid_info, WeightType.CENTER, sim_params)
    D = index_conservative_var(U_cart, ConservativeIndex.DENSITY, n_spatial_dim)
    Tau = index_conservative_var(U_cart, ConservativeIndex.TAU, n_spatial_dim)
    z = Tau+guess+D
    z = np.maximum(z, 1e-12)
    v_mag_2 = flux_squared/np.power(z,2)
    v_mag_2 = np.minimum(v_mag_2, 1.0 - 1e-10)
    W2 = np.power(1-v_mag_2,-1)
    W = np.sqrt(W2)
    rho = D/W
    velocities = ((flux.T)/(W.T*z.T)).T
    assert(~np.any(np.isnan(rho)))
    assert(~np.any(np.isnan(velocities)))
    assert(~np.any(np.isnan(guess)))
    assert(np.all(rho>0.0)) 
    assert(np.all(guess>0.0))
    output[...,PrimitiveIndex.DENSITY.value] = rho
    output[...,PrimitiveIndex.X_VELOCITY.value:] = velocities
    output[...,PrimitiveIndex.PRESSURE.value] = guess
    return output
