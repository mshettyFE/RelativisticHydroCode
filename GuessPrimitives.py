from HydroCore import SimParams
import numpy.typing as npt
import numpy as np
from metrics.Metric import Metric, WhichCacheTensor 
from GridInfo import GridInfo, WeightType
from HelperFunctions import index_conservative_var, index_primitive_var, PrimitiveIndex, ConservativeIndex
from EquationOfState import *

# https://iopscience.iop.org/article/10.1086/498238/pdf

def pressure_finding_func(guess_pressure_log: npt.ArrayLike,
                              U_cart: npt.ArrayLike, 
                               metric: Metric,
                                sim_params: SimParams,
                                grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    # Implements p(\rho(\bar{p}))
    guess_pressure = np.exp(guess_pressure_log)
    boost = boost_guess(guess_pressure,U_cart, metric, sim_params, grid_info, n_spatial_dim)
    D = index_conservative_var(U_cart, ConservativeIndex.DENSITY, n_spatial_dim)
    epsilon = internal_energy_guess(guess_pressure, U_cart, boost, n_spatial_dim)
    rho = D/boost
    current_pressure = pressure_from_epsilon(sim_params, epsilon, rho)
    out = current_pressure-guess_pressure
    return out

def pressure_finding_func_der(guess_pressure_log: npt.ArrayLike,
                              U_cart: npt.ArrayLike, 
                               metric: Metric,
                                sim_params: SimParams,
                                grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    # Implements f'(p) = |v(p)|^{2} c_{s}^{2}-1
    # q =  ln(p)
    # dq =   1/p dp  => dp/dq = p
    # df/dy = df/dp dp/dy
    guess_pressure = np.exp(guess_pressure_log)
    v_mag_2 = three_vel_mag_squared(guess_pressure, U_cart, metric, sim_params, grid_info, n_spatial_dim)
    boost = boost_guess(guess_pressure,U_cart, metric, sim_params, grid_info, n_spatial_dim)
    D = index_conservative_var(U_cart,ConservativeIndex.DENSITY, n_spatial_dim)
    rho = D/boost
    c_s  = sound_speed(sim_params,guess_pressure,rho)
    output = (v_mag_2*np.power(c_s, 2)-1  )*guess_pressure
    return output

def internal_energy_guess(guess_pressure_log: npt.ArrayLike, U_cart: npt.ArrayLike,
                          boost: npt.ArrayLike, n_spatial_dim: int) -> npt.NDArray:
    guess_pressure = np.exp(guess_pressure_log)
    # (\tau+D(1-W)+p(1-W)^{2})
    tau = index_conservative_var(U_cart,ConservativeIndex.TAU, n_spatial_dim)
    D = index_conservative_var(U_cart,ConservativeIndex.DENSITY, n_spatial_dim)
    return (tau+D*(1-boost)+guess_pressure*(1-np.power(boost,2)))/(D* boost)

def boost_guess(guess_pressure_log,
                  U_cart, 
                    metric: Metric,
                    sim_params: SimParams,
                    grid_info: GridInfo,
                     n_spatial_dim):
    guess_pressure = np.exp(guess_pressure_log)
    v_mag_2 = three_vel_mag_squared(guess_pressure, U_cart, metric, sim_params, grid_info, n_spatial_dim)
    return np.power(1-v_mag_2,-0.5)

def three_vel_mag_squared(guess_pressure_log,
                  U_cart, 
                    metric: Metric,
                    sim_params: SimParams,
                    grid_info: GridInfo,
                     n_spatial_dim):
    guess_pressure = np.exp(guess_pressure_log)
    fluxes = U_cart[..., ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    flux_squared = metric.three_vector_mag(fluxes,grid_info, WeightType.CENTER, sim_params)
    tau = index_conservative_var(U_cart, ConservativeIndex.TAU, n_spatial_dim)
    D = index_conservative_var(U_cart, ConservativeIndex.DENSITY, n_spatial_dim)
    bot = np.power(tau+D+guess_pressure,2)
    assert((flux_squared<bot).all())
    return flux_squared/bot

def construct_primitives_from_guess(guess_pressure_log:npt.ArrayLike,
                                     U_cart: npt.ArrayLike, metric: Metric,
                                     sim_params: SimParams,
                                     grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    output = np.zeros(U_cart.shape)
    # guess_pressure = np.clip(np.exp(guess_pressure_log), a_min=1E-9, a_max=None)
    guess_pressure = np.exp(guess_pressure_log)
    print("Final",guess_pressure)
    flux  = U_cart[..., ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    D = U_cart[..., ConservativeIndex.DENSITY.value]
    boost = boost_guess(guess_pressure,U_cart, metric, sim_params, grid_info, n_spatial_dim)
    rho =  D/boost
    output[..., PrimitiveIndex.DENSITY.value] = rho
    epsilon = internal_energy_guess(guess_pressure, U_cart, boost, n_spatial_dim)
    enthalphy = internal_enthalpy_from_internal_energy(epsilon, sim_params)
    output[...,PrimitiveIndex.X_VELOCITY.value:] = ((flux.T)/(rho*enthalphy*boost).T).T
    output[...,PrimitiveIndex.PRESSURE.value] = guess_pressure
    return output