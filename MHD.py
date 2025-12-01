import numpy as np 
from enum import Enum
import numpy.typing as npt
import pickle as pkl
from HydroCore import PrimitiveIndex, SimParams, SimulationState
from UpdateSteps import SpatialUpdateType, SpatialUpdate,TimeUpdateType
from  GridInfo import GridInfo, WeightType, CoordinateChoice
from BoundaryManager import BoundaryConditionManager, BoundaryCondition
import Plotting

def save_results(
        history: list[tuple[np.float64, npt.NDArray]],
        sim_state: SimulationState,
        filename: str = "snapshot.pkl"
        ):
    data = (history, sim_state)
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

def SodShockInitialization(rho_l: np.float64, v_l: np.float64, P_l: np.float64,
                           rho_r:np.float64, v_r: np.float64, P_r:np.float64,
                           N_cells: np.int64 = 10000,
                           t_max: np.float64 = 0.2) -> SimulationState:
    grid_info = GridInfo(np.array([0.0]), np.array([1.0]), np.array([N_cells]))
    spatial_update = SpatialUpdate(SpatialUpdateType.FLAT, {})
    simulation_params = SimParams(1.4, 0.5, t_max, 1.0, coordinate_system= CoordinateChoice.CARTESIAN,
                                  include_source=False, time_integration=TimeUpdateType.RK3, spatial_integration=spatial_update)
    bcm = BoundaryConditionManager([BoundaryCondition.ZERO_GRAD], [BoundaryCondition.ZERO_GRAD])
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    assert(primitives.ndim==2)
    grid_centers = grid_info.construct_grid_centers(0)
    lower_half = grid_centers<0.5 
    upper_half = grid_centers>=0.5
    primitives[lower_half, PrimitiveIndex.DENSITY.value] = rho_l
    primitives[lower_half, PrimitiveIndex.PRESSURE.value] = P_l
    primitives[lower_half, PrimitiveIndex.X_VELOCITY.value] = v_l
    primitives[upper_half, PrimitiveIndex.DENSITY.value] = rho_r
    primitives[upper_half, PrimitiveIndex.PRESSURE.value] = P_r
    primitives[upper_half, PrimitiveIndex.X_VELOCITY.value] = v_r
    return SimulationState(primitives, grid_info, bcm, simulation_params)

def BondiAccretionInitialization(
        rho: np.float64,
        v: np.float64,
        P: np.float64,
        N_cells: np.float64
    ):
    grid_info = GridInfo(np.array([0.5]), np.array([10.15]), np.array([N_cells]))
    spatial_update = SpatialUpdate(SpatialUpdateType.FLAT, {"theta": 1.5})
    simulation_params = SimParams(1.4, 0.2, 20.0,1.0, coordinate_system= CoordinateChoice.SPHERICAL, 
                                  include_source=True, time_integration=TimeUpdateType.EULER  , spatial_integration=spatial_update) 
    bcm = BoundaryConditionManager([BoundaryCondition.ZERO_GRAD], [BoundaryCondition.FIXED])
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    primitives[:, PrimitiveIndex.DENSITY.value] = rho
    primitives[:, PrimitiveIndex.X_VELOCITY.value] = v
    primitives[:, PrimitiveIndex.PRESSURE.value] = P
    return SimulationState(primitives, grid_info, bcm, simulation_params)

class WhichTestProblem(Enum):
    CARTESIAN_SOD=0 
    HARDER_SOD=1 
    BONDI_PROBLEM=2

def runSim(which_sim: WhichTestProblem):
    match which_sim:
        case WhichTestProblem.CARTESIAN_SOD:
            save_frequency = 1
            state_sim =  SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2) 
        case WhichTestProblem.HARDER_SOD:
            state_sim = SodShockInitialization(10.0,0.0,100.0, 1.0, 0.0, 1.0, N_cells=1000, t_max=0.1)
            save_frequency = 100
        case WhichTestProblem.BONDI_PROBLEM:
            state_sim = BondiAccretionInitialization(1.0, 0.0, 0.1, 100)
            save_frequency = 1
        case _:
            raise Exception("Unimplemented test problem")
    history = []
    iteration = 0
    while(state_sim.current_time < state_sim.simulation_params.t_max):
        t, state = state_sim.update()
        if(iteration%save_frequency==0):
            history.append( (t,state))
        iteration += 1
        print(t, state_sim.simulation_params.t_max)
    save_results(history, state_sim)

def peak_finder(
    input_pkl_file: str = "snapshot.pkl",
    padding:int =100
    ):
    assert(padding%2==0)
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    U_cartesian = params.grid_info.unweight_vector(history[1][-1], params.simulation_params.coordinate_system, WeightType.CENTER)
    W = params.conservative_to_primitive(  U_cartesian)
    dx = params.grid_info.delta()
    der_prof = (W[1:,:]-W[:-1,:])/(dx) 
    rho_der= der_prof[:, PrimitiveIndex.DENSITY.value]
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
#     runSim(WhichTestProblem.CARTESIAN_SOD)
#    Plotting.plot_results_1D()
#    peak_finder()
#    runSim(WhichTestProblem.HARDER_SOD)
#    Plotting.plot_results_1D()
    runSim(WhichTestProblem.BONDI_PROBLEM)
    Plotting.plot_Mdot_time("snapshot.pkl")
    Plotting.plot_Mdot_position("snapshot.pkl")
    Plotting.plot_results_1D("snapshot.pkl",title="Bondi Accretion", filename="BondiAccretion.png", xlabel="r", show_mach=True)
