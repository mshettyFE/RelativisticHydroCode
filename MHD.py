import numpy as np 
import numpy.typing as npt
import pickle as pkl
from HydroCore import *
from Parameters import SimParams, InitialParameters
from UpdateSteps import SpatialUpdateType, SpatialUpdate,TimeUpdateType
import GridInfo
from BoundaryManager import BoundaryConditionManager, BoundaryCondition
import Plotting

def save_results(
        history: list[tuple[np.float64, npt.NDArray]],
        params: Parameters.InitialParameters,
        filename: str = "snapshot.pkl"
        ):
    data = (history, params)
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

def SodShockInitialization(rho_l: np.float64, v_l: np.float64, P_l: np.float64,
                           rho_r:np.float64, v_r: np.float64, P_r:np.float64,
                           N_cells: np.int64 = 10000,
                           t_max: np.float64 = 0.2) -> InitialParameters:
    grid_info = GridInfo.GridInfo(np.array([0.0]), np.array([1.0]), np.array([N_cells]))
    spatial_update = SpatialUpdate(SpatialUpdateType.FLAT, {})
    simulation_params = SimParams(1.4, 0.5, t_max, 1.0, GridInfo.CoordinateChoice.CARTESIAN,
                                  include_source=False, time_integration=TimeUpdateType.RK3, spatial_integration=spatial_update)
    bcm = BoundaryConditionManager([BoundaryCondition.ZERO_GRAD], [BoundaryCondition.ZERO_GRAD])
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    grid_centers = grid_info.construct_grid_centers(0)
    lower_half = grid_centers<0.5 
    upper_half = grid_centers>=0.5
    primitives[lower_half, PrimativeIndex.DENSITY.value] = rho_l
    primitives[lower_half, PrimativeIndex.VELOCITY.value] = v_l
    primitives[lower_half, PrimativeIndex.PRESSURE.value] = P_l
    primitives[upper_half, PrimativeIndex.DENSITY.value] = rho_r
    primitives[upper_half, PrimativeIndex.VELOCITY.value] = v_r
    primitives[upper_half, PrimativeIndex.PRESSURE.value] = P_r
    initial_conds = primitive_to_conservative(primitives, simulation_params)
    return InitialParameters(grid_info, simulation_params, bcm, initial_conds)

def M_dot(U_profiles: npt.ArrayLike, params: InitialParameters):
    r_center = params.grid_info.construct_grid_centers(0) 
    W = conservative_to_primitive(U_profiles, params)
    return W[:, PrimativeIndex.DENSITY.value]*W[:,PrimativeIndex.VELOCITY.value]*np.power(r_center,2)

def BondiAccretionInitialization(
        rho: np.float64,
        v: np.float64,
        P: np.float64,
        N_cells: np.float64
    ):
    grid_info = GridInfo.GridInfo(np.array([0.1]), np.array([10.1]), np.array([N_cells]))
    spatial_update = SpatialUpdate(SpatialUpdateType.FLAT, {"theta": 1.5})
    simulation_params = SimParams(1.4, 0.2, 20.0,1.0, GridInfo.CoordinateChoice.SPHERICAL, 
                                  include_source=True, time_integration=TimeUpdateType.EULER  , spatial_integration=spatial_update) 
    bcm = BoundaryConditionManager([BoundaryCondition.ZERO_GRAD], [BoundaryCondition.FIXED])
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    primitives[:, PrimativeIndex.DENSITY.value] = rho
    primitives[:, PrimativeIndex.VELOCITY.value] = v
    primitives[:, PrimativeIndex.PRESSURE.value] = P
    initial_conds = primitive_to_conservative(primitives, simulation_params)
    return InitialParameters(grid_info, simulation_params, bcm, initial_conds)

def CartesianSodProblem():
    initial_params = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2)
    state = initial_params.initial_weighted_U
    current_time = 0.0 
    history = []
    while(current_time < initial_params.simulation_params.t_max):
        updated_state = update(state, current_time, initial_params)
        history.append(updated_state)
        current_time, state = updated_state
    save_results(history,initial_params)

def HarderSodProblem():
    initial_params = SodShockInitialization(10.0,0.0,100.0, 1.0, 0.0, 1.0, t_max=0.1)
    state = initial_params.initial_weighted_U
    current_time = 0.0 
    history = []
    iteration  = 0
    while(current_time < initial_params.simulation_params.t_max):
        updated_state = update(state, current_time, initial_params)
        if(iteration%100==0):
            #print(iteration, current_time, initial_params.simulation_params.t_max)
            history.append(updated_state)
        history.append(updated_state)
        current_time, state = updated_state
        iteration += 1
    save_results(history,initial_params)

def BondiAccretionProblem():
    initial_params = BondiAccretionInitialization(1.0, 0.0, 0.1, 100)
    state = initial_params.initial_weighted_U 
    current_time = 0.0
    history = []
    iteration = 0
    while(current_time < initial_params.simulation_params.t_max):
        #print(iteration, current_time, initial_params.simulation_params.t_max)
        updated_state = update(state, current_time, initial_params)
        history.append(updated_state)
        current_time, state = updated_state
        iteration += 1
    save_results(history,initial_params) 

def peak_finder(
    input_pkl_file: str = "snapshot.pkl",
    padding:int =100
    ):
    assert(padding%2==0)
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    W = conservative_to_primitive(history[-1][1], params)
    dx = params.grid_info.delta()
    der_prof = (W[1:,:]-W[:-1,:])/(dx) 
    rho_der= der_prof[:, PrimativeIndex.DENSITY.value]
    rho_der_right_pad = np.insert(rho_der, rho_der.shape[0], np.full(padding, -np.inf))
    rho_der_pad = np.insert(rho_der_right_pad, 0, np.full(padding, -np.inf))
    mask = np.full(rho_der_pad.shape, True) 
    for shift in range(-padding,padding+1):
        if(shift==0):
            continue
        shifted_rho_der_pad = np.roll(rho_der_pad, shift)
        new_mask = (rho_der_pad < shifted_rho_der_pad)
        mask &= new_mask 
    unpadded_mask = mask[padding:-padding]
    support = params.grid_info.construct_grid_edges(0)[1:-1]
    
if __name__ == "__main__":
   CartesianSodProblem()
   Plotting.plot_results()
#    peak_finder()
#    HarderSodProblem()
    # Plotting.plot_results("snapshot.pkl")
    # BondiAccretionProblem()
    # Plotting.plot_Mdot_time("snapshot.pkl")
    # #Plotting.plot_Mdot_position("snapshot.pkl")
    # Plotting.plot_results("snapshot.pkl",title="Bondi Accretion", filename="BondiAccretion.png", xlabel="r", show_mach=True)
