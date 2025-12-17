from HydroCore import SimParams
import numpy.typing as npt
import numpy as np
from metrics.Metric import Metric, WhichCacheTensor 
from GridInfo import GridInfo, WeightType
from HelperFunctions import index_conservative_var, index_primitive_var, PrimitiveIndex, ConservativeIndex
from EquationOfState import *

# https://iopscience.iop.org/article/10.1086/498238/pdf
# guess = \rho h boost^2

def root_finding_func(guess: npt.ArrayLike,
                              U_cart: npt.ArrayLike, 
                               metric: Metric,
                                sim_params: SimParams,
                                grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    flux = U_cart[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    alpha = metric.get_metric_product(grid_info, WhichCacheTensor.ALPHA, WeightType.CENTER, sim_params)
    flux_squared = metric.three_vector_mag(flux, grid_info, WeightType.CENTER, sim_params)
    v_mag_2 = flux_squared/np.power(guess,2)
    # v_mag_2 = np.clip(v_mag_2, 0, 1.0 - 1e-14)    
    boost = np.power(1-v_mag_2,-0.5)
    D = index_conservative_var(U_cart, ConservativeIndex.DENSITY, n_spatial_dim)
    Tau = index_conservative_var(U_cart, ConservativeIndex.TAU, n_spatial_dim)
    rho = D/boost
    P = guess-Tau-D
    epsilon = guess/(D*boost)-1-P/rho
    guess_pressure = equation_of_state_epsilon(sim_params, epsilon,rho )
    out  =guess-guess_pressure-D-Tau
    return out

def root_finding_func_der(guess: npt.ArrayLike,
                              U_cart: npt.ArrayLike, 
                               metric: Metric,
                                sim_params: SimParams,
                                grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    # Assumes ideal gas equation of state
    flux = U_cart[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    alpha = metric.get_metric_product(grid_info, WhichCacheTensor.ALPHA, WeightType.CENTER, sim_params)
    flux_squared = metric.three_vector_mag(flux, grid_info, WeightType.CENTER, sim_params)
    v_mag_2 = flux_squared/np.power(guess,2)
    boost = np.power(1-v_mag_2,-0.5)
    D = index_conservative_var(U_cart, ConservativeIndex.DENSITY, n_spatial_dim)
    rho = D/boost
    Tau = index_conservative_var(U_cart, ConservativeIndex.TAU, n_spatial_dim)
    P = guess-Tau-D
    epsilon = guess/(D*boost)-1-P/rho
    dP_drho = (sim_params.gamma-1)*epsilon
    dP_depsilon = (sim_params.gamma-1)*rho
    drho_dz = D*v_mag_2/guess 
    depsilon_dz = boost/sim_params.gamma/D
    out = 1-dP_drho*drho_dz-dP_depsilon*depsilon_dz
    return out


def construct_primitives_from_guess(guess:npt.ArrayLike,
                                     U_cart: npt.ArrayLike, metric: Metric,
                                     sim_params: SimParams,
                                     grid_info: GridInfo,
                                n_spatial_dim: int) -> npt.NDArray:
    output = np.zeros(U_cart.shape)
    flux = U_cart[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:]
    alpha = metric.get_metric_product(grid_info, WhichCacheTensor.ALPHA, WeightType.CENTER, sim_params)
    flux_squared = metric.three_vector_mag(flux, grid_info, WeightType.CENTER, sim_params)
    v_mag_2 = flux_squared/np.power(guess,2)
    boost = np.power(1-v_mag_2,-0.5)
    D = index_conservative_var(U_cart, ConservativeIndex.DENSITY, n_spatial_dim)
    Tau = index_conservative_var(U_cart, ConservativeIndex.TAU, n_spatial_dim)
    rho = D/boost
    P = guess-Tau-D
    output[..., PrimitiveIndex.X_VELOCITY.value:] = (flux.T/guess.T).T
    output[...,PrimitiveIndex.DENSITY.value] = rho 
    output[...,PrimitiveIndex.PRESSURE.value] = P
    return output