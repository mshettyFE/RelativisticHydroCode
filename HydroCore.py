import numpy.typing as npt
from enum import Enum
import numpy as np
import pickle as pkl
from BoundaryManager import BoundaryCondition 
from GridInfo import WeightType
from UpdateSteps import TimeUpdateType, SpatialUpdateType
import Parameters

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
    rho = U[:,ConservativeIndex.DENSITY.value]
    assert(np.all(rho!=0))
    v = U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/rho
    E = U[:,ConservativeIndex.ENERGY.value]
    # Total energy
    #   E = \rho *e + 0.5*\rho * v**2
    # Inverting 
    #  E/rho-0.5 v**2 = e
    e = E/rho-0.5*np.power(v,2)
    return e

def internal_energy_primitive(W: npt.ArrayLike, simulation_params: Parameters.SimParams) -> npt.NDArray[np.float64]:
    # primitive variables 
    # W = (\rho, v, P)
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # Invert this: e = P/((\gamma-1) * \rho)
    return W[:,PrimativeIndex.PRESSURE.value]/( (simulation_params.gamma-1) * W[:,PrimativeIndex.DENSITY.value])

def equation_of_state_conservative(U: npt.ArrayLike, global_parameters: Parameters.InitialParameters) -> npt.NDArray[np.float64]:
    # Return pressure given the conservative variables 
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # e is the internal energy
    # gamma is adiabatic index 
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    e = internal_energy_conservative(U)
    assert np.all(e>=0)
    return (global_parameters.simulation_params.gamma-1)*U[:,ConservativeIndex.DENSITY.value]*e

def conservative_to_primitive(U: npt.ArrayLike, global_parameters: Parameters.InitialParameters) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = U[:,ConservativeIndex.DENSITY.value]
    pressure = equation_of_state_conservative(U, global_parameters)
    assert(np.all(rho!=0))
    return np.stack([rho, U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/ rho, pressure], axis=1) 

def primitive_to_conservative(W: npt.ArrayLike, sim_params: Parameters.SimParams) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = W[:,PrimativeIndex.DENSITY.value]
    velocity = W[:,PrimativeIndex.VELOCITY.value]
    # Total energy
    #   E = \rho *e + 0.5*\rho * v**2
    E = rho * internal_energy_primitive(W, sim_params) + 0.5*rho*np.power(velocity,2)
    return np.stack([rho, rho*velocity, E], axis= 1) 

def flux_from_conservative(U: npt.ArrayLike, global_parameters: Parameters.InitialParameters) -> npt.NDArray[np.float64]:
    primitive = conservative_to_primitive(U,global_parameters)
    F_0 = U[:,ConservativeIndex.MOMENTUM_DENSITY.value]
    F_1 = F_0*primitive[:,PrimativeIndex.VELOCITY.value]+primitive[:,PrimativeIndex.PRESSURE.value]
    F_2 = (U[:,ConservativeIndex.ENERGY.value]+primitive[:,PrimativeIndex.PRESSURE.value])*primitive[:,PrimativeIndex.VELOCITY.value]
    return np.stack([F_0,F_1,F_2], axis=1)

def sound_speed(W: npt.ArrayLike, global_parameters: Parameters.InitialParameters):
    return np.sqrt(global_parameters.simulation_params.gamma*W[:, PrimativeIndex.PRESSURE.value]/ W[:, PrimativeIndex.DENSITY.value])

def alpha_plus_minus(U_padded: npt.ArrayLike, global_parameters: Parameters.InitialParameters) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    U = U_padded
    primitives = conservative_to_primitive(U, global_parameters)
    c_s = sound_speed(primitives, global_parameters)
    lambda_plus = primitives[:,PrimativeIndex.VELOCITY.value]+c_s
    lambda_minus = primitives[:,PrimativeIndex.VELOCITY.value]-c_s
    # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for + and MAX(0,-left, -right) for - 
    zeros = np.zeros(lambda_plus.shape[0]-2)
    # First, the plus case   
    # Grab the left and right speeds from the padded array
    alpha_plus_candidates = np.stack([zeros, lambda_plus[:-2], lambda_plus[2:]], axis=1)    
    # Calculate max across each row
    alpha_plus = np.max(alpha_plus_candidates, axis=1)
    # Dont forget the minus sign prefactors here! (GPT caught this...) 
    # Prompt: Pasted in source file and stated that internal energy was going negative after 1 timestep; please point out potential errors 
    # It gave me some B.S. about Rusanov flux (which I'm not implementing) and incorrect signage of time update (which is wrong)
    # It did catch the sign error though (which I forgot to add after reading the hydro PDF)
    alpha_minus_candidates = np.stack([zeros, -lambda_minus[:-2], -lambda_minus[2:]], axis=1)    
    alpha_minus = np.max(alpha_minus_candidates, axis=1)
    return (alpha_minus, alpha_plus)

def spatial_derivative(U_padded: npt.ArrayLike, global_parameters: Parameters.InitialParameters, spatial_index: np.uint = 0) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # Assuming that U is Cartesian. Cell_scaling is for the fluxes
    cell_flux = flux_from_conservative(U_padded, global_parameters)
    alpha_minus, alpha_plus = alpha_plus_minus(U_padded, global_parameters)
    alpha_sum = alpha_minus+alpha_plus 
    assert(np.all(alpha_sum != 0))
    alpha_prod = alpha_minus*alpha_plus
    # Augment flux array as well with 0 flux conditions 
    # Bunch of .T because numpy broadcasting rules
    left_cell_flux = cell_flux[1:-1,:].T 
    left_conserve = U_padded[1:-1,:].T
    right_cell_flux =cell_flux[2:,:].T  
    right_conserve = U_padded[2:,:].T
    cell_flux_plus_half = (alpha_plus*left_cell_flux+ alpha_minus*right_cell_flux-alpha_prod*(right_conserve-left_conserve))/alpha_sum 
    # Now the minus branch
    left_cell_flux = cell_flux[:-2,:].T 
    left_conserve = U_padded[:-2,:].T
    right_cell_flux =cell_flux[1:-1,:].T  
    right_conserve = U_padded[1:-1,:].T
    cell_flux_minus_half = (alpha_plus*left_cell_flux+ alpha_minus*right_cell_flux-alpha_prod*(right_conserve-left_conserve))/alpha_sum 
    weights = global_parameters.grid_info.weights(global_parameters.simulation_params.coordinate_system, WeightType.EDGE) 
    cell_flux_plus_half_rescaled = cell_flux_plus_half* weights[1:]
    cell_flux_minus_half_rescaled = cell_flux_minus_half* weights[:-1]
    return -(cell_flux_plus_half_rescaled.T-cell_flux_minus_half_rescaled.T)/global_parameters.grid_info.delta()[spatial_index], alpha_plus, alpha_minus

def calc_dt(alpha_plus: npt.ArrayLike, alpha_minus:npt.ArrayLike, global_parameters: Parameters.InitialParameters):
    max_alpha = np.max( [alpha_plus, alpha_minus]) 
    return global_parameters.simulation_params.Courant*np.min(global_parameters.grid_info.delta())/max_alpha

def SourceTerm(U:npt.ArrayLike, initial_params: Parameters.InitialParameters):
    # Assumes that U is the array of Cartesian conservative variables
    W = conservative_to_primitive(U, initial_params)
    grid_centers = initial_params.grid_info.construct_grid_centers(0)
    # primitive variables 
    # W = (\rho, v, P)
    S_0 = np.zeros(grid_centers.shape)
    S_1 = 1.0*(2*W[:,PrimativeIndex.PRESSURE.value]*grid_centers-W[:,PrimativeIndex.DENSITY.value]*initial_params.simulation_params.GM)
    S_2 = 1.0*(-W[:,PrimativeIndex.DENSITY.value]*W[:,PrimativeIndex.VELOCITY.value]*initial_params.simulation_params.GM) 
    return np.stack([S_0,S_1,S_2], axis=1) 

def minmod(x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike):
    sgn_x = np.sign(x)
    sgn_y = np.sign(y)
    sgn_z = np.sign(z)
    min_candidates = np.min(np.abs(np.stack([x,y,z], axis=1)), axis=1)
    output = (1/4)*np.abs(sgn_x+sgn_y)*(sgn_x+sgn_z)*min_candidates
    assert(output.shape == x.shape)
    return output

def LinearUpdate(U_scaled: npt.ArrayLike, global_parameters: Parameters.InitialParameters): 
    match global_parameters.simulation_params.spatial_integration.method:
        case SpatialUpdateType.FLAT:
            U_padded = pad_array(U_scaled, global_parameters, 1)
            flux_change, alpha_plus, alpha_minus = spatial_derivative(U_padded, global_parameters, 0)
            dt = calc_dt(alpha_plus, alpha_minus, global_parameters)
            state_update = flux_change
            if(global_parameters.simulation_params.include_source):
                state_update += SourceTerm(U_padded[1:-1], global_parameters)
            return dt, state_update
        case SpatialUpdateType.PLM:
            U_padded = pad_array(U_scaled, global_parameters, 2)
            theta = global_parameters.simulation_params.spatial_update.params["theta"]
            left_interface = U_padded + 0.5*minmod(
                    theta*(U_padded[2:-2,:]-U_padded[1:-3,:]), 
                    0.5*(U_padded[3:-1,:]-U_padded[1:-3,:]),
                    theta*(U_padded[3:-1,:]-U_padded[2:-2,:])
            )
            raise Exception("Unimplemented Spatial Update")
        case _:
            raise Exception("Unimplemented Spatial Update")

def update(U: npt.ArrayLike, current_time: np.float64, global_parameters: Parameters.InitialParameters) -> tuple[np.float64, npt.NDArray]:
    assert(U.shape== global_parameters.initial_weighted_U.shape)
    # Undo scaling for input
    U_scaled = global_parameters.grid_info.unweight_vector(U, global_parameters.simulation_params.coordinate_system, WeightType.CENTER)
    dt, state_update_1 = LinearUpdate(U_scaled, global_parameters)
    U_1 = U+dt*state_update_1

    match global_parameters.simulation_params.time_integration:
        case TimeUpdateType.EULER:
            return current_time+dt, U_1
        case TimeUpdateType.RK3:
            U_scaled_1 = global_parameters.grid_info.unweight_vector(U_1, global_parameters.simulation_params.coordinate_system, WeightType.CENTER)
            _, state_update_2 = LinearUpdate(U_scaled_1, global_parameters)
            U_2 = (3/4)*U+(1/4)*U_1+(1/4)*dt*state_update_2 
            U_scaled_2 = global_parameters.grid_info.unweight_vector(U_2, global_parameters.simulation_params.coordinate_system, WeightType.CENTER)
            _, state_update_3 = LinearUpdate(U_scaled_2, global_parameters)
            return current_time+dt, (1/3)*U+(2/3)*U_2+(2/3)*dt*state_update_3
        case _:
            raise Exception("Unimplemented TimeUpdateType Method")
 
def pad_array(var:npt.ArrayLike,global_parameters: Parameters.InitialParameters, n_cells = 1, spatial_index=0):
    # Augment the array to incorporate the BCs
    # Assuming that var is Cartesian
    # Wastful to do this every time, but it's fine...
    var_initial = global_parameters.grid_info.unweight_vector(global_parameters.initial_weighted_U, global_parameters.simulation_params.coordinate_system, WeightType.CENTER) 
    zero_index_row = np.tile(var[0,:], (n_cells,1)) 
    last_index_row  = np.tile(var[-1,:], (n_cells,1))
    zero_index_row_initial = np.tile(var_initial[0,:], (n_cells,1)) 
    last_index_row_initial  = np.tile(var_initial[-1,:], (n_cells,1))
    left_bc, right_bc = global_parameters.bcm.get_boundary_conds(spatial_index)
    match right_bc:
        case BoundaryCondition.ZERO_GRAD:
            var_right_pad = np.insert(var, var.shape[0],last_index_row , axis=0)
        case BoundaryCondition.FIXED:
            var_right_pad = np.insert(var, var.shape[0],last_index_row_initial, axis=0)
        case _:
            raise Exception("Unimplemented BC")
    match left_bc:
        case BoundaryCondition.ZERO_GRAD:
            var_padded  = np.insert(var_right_pad, 0, zero_index_row, axis=0) 
        case BoundaryCondition.FIXED:
            var_padded  = np.insert(var_right_pad, 0, zero_index_row_initial, axis=0) 
        case _:
            raise Exception("Unimplemented BC")
    return var_padded    

if __name__ == "__main__":
    pass
