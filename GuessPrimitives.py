from HydroCore import SimParams
import numpy.typing as npt
import numpy as np
from metrics.Metric import Metric, WhichCacheTensor 
from GridInfo import GridInfo, WeightType
from HelperFunctions import index_conservative_var, index_primitive_var, PrimitiveIndex, ConservativeIndex
from EquationOfState import *

def pressure_finding_function(guess_pressure: npt.ArrayLike,
                              U_cart: npt.ArrayLike, 
                               metric: Metric,
                                sim_params: SimParams,
                                grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    # Implements p(\rho(\bar{p}))
    vels = velocity_guess(guess_pressure, U_cart, n_spatial_dim)
    alpha  = metric.get_metric_product(grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, sim_params) 
    boost =     metric.boost_field(alpha, vels, grid_info , WeightType.CENTER, sim_params)
    D = index_conservative_var(U_cart,ConservativeIndex.DENSITY, n_spatial_dim)    
    rho = density_guess(D, boost)
    epsilon = internal_energy_guess(guess_pressure, U_cart, boost, n_spatial_dim)
    out = pressure_from_epsilon(sim_params, epsilon, rho)-guess_pressure
    return out

def pressure_finding_func_der(guess_pressure, 
                               U_cart: npt.ArrayLike, 
                               metric: Metric,
                                sim_params: SimParams,
                                grid_info: GridInfo, 
                                n_spatial_dim: int) -> npt.NDArray:
    # Implements f'(p) = |v(p)|^{2} c_{s}^{2}-1
    vels = velocity_guess(guess_pressure, U_cart, n_spatial_dim)
    vel_magnitude = metric.spatial_vel_mag(vels, grid_info, WeightType.CENTER, sim_params)
    alpha  = metric.get_metric_product(grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, sim_params) 
    boost =     metric.boost_field(alpha, vels, grid_info , WeightType.CENTER, sim_params) # NOTE: implicit Duplicate |v|^{2} 
    D = index_conservative_var(U_cart,ConservativeIndex.DENSITY, n_spatial_dim)    
    rho = density_guess(D, boost)
    speed_sound = sound_speed(sim_params, guess_pressure, rho)
    return vel_magnitude*np.power(speed_sound,2)-1

def velocity_guess(guess_pressure: npt.ArrayLike, U_cart: npt.ArrayLike,
                                n_spatial_dim: int) -> npt.NDArray:
    # v^{i} = S^{i}/(D+\tau+guess_p)
    momenta = U_cart[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    tau = index_conservative_var(U_cart,ConservativeIndex.TAU, n_spatial_dim)
    D = index_conservative_var(U_cart,ConservativeIndex.DENSITY, n_spatial_dim)
    assert(tau.shape==guess_pressure.shape)
    return (momenta.T/(tau+D+guess_pressure).T).T

def internal_energy_guess(guess_pressure: npt.ArrayLike, U_cart: npt.ArrayLike,
                          boost: npt.ArrayLike, n_spatial_dim: int) -> npt.NDArray:
    # (\tau+D(1-W)+p(1-W)^{2})
    tau = index_conservative_var(U_cart,ConservativeIndex.TAU, n_spatial_dim)
    D = index_conservative_var(U_cart,ConservativeIndex.DENSITY, n_spatial_dim)
    return (tau+D*(1-boost)+guess_pressure*(1-np.power(boost,2)))/(D* boost)

def density_guess( D: npt.ArrayLike, boost: npt.ArrayLike) -> npt.NDArray:
    return D/boost

def construct_primitives_from_guess(guess_pressure:npt.ArrayLike,
                                     U_cart: npt.ArrayLike, metric: Metric,
                                     sim_params: SimParams,
                                     grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    vels = velocity_guess(guess_pressure, U_cart, n_spatial_dim)
    alpha  = metric.get_metric_product(grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, sim_params) 
    boost =     metric.boost_field(alpha, vels, grid_info , WeightType.CENTER, sim_params)    
    D = index_conservative_var(U_cart,ConservativeIndex.DENSITY, n_spatial_dim)
    rho = density_guess(D, boost)
    output = np.zeros(U_cart.shape)
    output[..., PrimitiveIndex.DENSITY.value] = rho
    output[..., PrimitiveIndex.PRESSURE.value ] = guess_pressure 
    output[..., PrimitiveIndex.X_VELOCITY.value:] = vels 
    return output