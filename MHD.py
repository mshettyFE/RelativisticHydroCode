import numpy as np 
from enum import Enum
import numpy.typing as npt
import pickle as pkl
from HydroCore import PrimitiveIndex, SimParams, SimulationState
from UpdateSteps import SpatialUpdateType, SpatialUpdate,TimeUpdateType
from  GridInfo import GridInfo
from BoundaryManager import BoundaryConditionManager, BoundaryCondition
import Plotting
from metrics.CartesianMinkowski_1_1 import CartesianMinkowski_1_1
from metrics.SphericalMinkowski_1_3 import SphericalnMinkowski_1_3
from metrics.CartesianMinkowski_1_2 import CartesianMinkowski_1_2

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
    simulation_params = SimParams(1.4, 0.5, t_max, 1.0,
                                  include_source=False, time_integration=TimeUpdateType.RK3, spatial_integration=spatial_update)
    bcm = BoundaryConditionManager([BoundaryCondition.ZERO_GRAD], [BoundaryCondition.ZERO_GRAD])
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    assert(primitives.ndim==2)
    metric = CartesianMinkowski_1_1(grid_info)
    assert(metric.dimension==2) # 1+1 
    grid_centers = grid_info.construct_grid_centers(0)
    lower_half = grid_centers<0.5 
    upper_half = grid_centers>=0.5
    primitives[lower_half, PrimitiveIndex.DENSITY.value] = rho_l
    primitives[lower_half, PrimitiveIndex.PRESSURE.value] = P_l
    primitives[lower_half, PrimitiveIndex.X_VELOCITY.value] = v_l
    primitives[upper_half, PrimitiveIndex.DENSITY.value] = rho_r
    primitives[upper_half, PrimitiveIndex.PRESSURE.value] = P_r
    primitives[upper_half, PrimitiveIndex.X_VELOCITY.value] = v_r
    return SimulationState(
        primitives,grid_info, bcm, simulation_params, metric
    )

def BondiAccretionInitialization(
        rho: np.float64,
        v: np.float64,
        P: np.float64,
        N_cells: np.float64,
        t_max: np.float64 = 5
    ):
    grid_info = GridInfo(np.array([0.5,np.pi/2,0]), np.array([10.15,np.pi/2,0]), np.array([N_cells,1,1]))
    spatial_update = SpatialUpdate(SpatialUpdateType.FLAT, {"theta": 1.5})
    simulation_params = SimParams(1.4, 0.2, t_max,1.0,  
                                  include_source=True, time_integration=TimeUpdateType.EULER  , spatial_integration=spatial_update) 
    bcm = BoundaryConditionManager(
        [BoundaryCondition.ZERO_GRAD,BoundaryCondition.ZERO_GRAD,BoundaryCondition.ZERO_GRAD], [BoundaryCondition.FIXED,BoundaryCondition.ZERO_GRAD,BoundaryCondition.ZERO_GRAD]
        )
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[5]  ) 
    primitives[..., PrimitiveIndex.DENSITY.value] = rho
    primitives[..., PrimitiveIndex.X_VELOCITY.value] = v # r
    primitives[..., PrimitiveIndex.Y_VELOCITY.value] = 0 # theta
    primitives[..., PrimitiveIndex.Z_VELOCITY.value] = 0 # phi
    primitives[..., PrimitiveIndex.PRESSURE.value] = P
    metric = SphericalnMinkowski_1_3(grid_info)
    out = SimulationState(
        primitives,grid_info, bcm, simulation_params, metric
    )
    return out
class Which1DTestProblem(Enum):
    CARTESIAN_SOD=0 
    HARDER_SOD=1 
    BONDI_PROBLEM=2

def runSim1D(which_sim: Which1DTestProblem):
    match which_sim:
        case Which1DTestProblem.CARTESIAN_SOD:
            save_frequency = 1
            which_axes = ()
            state_sim =  SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2) 
        case Which1DTestProblem.HARDER_SOD:
            state_sim = SodShockInitialization(10.0,0.0,100.0, 1.0, 0.0, 1.0, N_cells=1000, t_max=0.1)
            save_frequency = 100
            which_axes = ()
        case Which1DTestProblem.BONDI_PROBLEM:
            state_sim = BondiAccretionInitialization(1.0, 0.0, 0.1, 100, t_max=10)
            save_frequency = 1
            which_axes = tuple([0]) # Only evolve along r coordinate
        case _:
            raise Exception("Unimplemented test problem")
    history = []
    iteration = 0
    while(state_sim.current_time < state_sim.simulation_params.t_max):
        t, state = state_sim.update(which_axes)
        if(iteration%save_frequency==0):
            history.append( (t,state))
        iteration += 1
        print(t, state_sim.simulation_params.t_max)
    save_results(history, state_sim)

def ImplosionInitialization(t_max = 2.5, N_cells = 100):
    grid_info = GridInfo(np.array([0.0,0.0]), np.array([0.3,0.3]), np.array([N_cells,N_cells]))
    spatial_update = SpatialUpdate(SpatialUpdateType.FLAT, {})
    simulation_params = SimParams(1.4, 0.5, t_max, 1.0,
                                  include_source=False, time_integration=TimeUpdateType.RK3, spatial_integration=spatial_update)
    bcm = BoundaryConditionManager(
            [BoundaryCondition.ZERO_GRAD, BoundaryCondition.ZERO_GRAD], 
            [BoundaryCondition.ZERO_GRAD, BoundaryCondition.ZERO_GRAD]
            )
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[4]  ) 
    grid_centers_x = grid_info.construct_grid_centers(0)
    grid_centers_y = grid_info.construct_grid_centers(1)
    xx,yy  = np.meshgrid(grid_centers_x, grid_centers_y)
    summed = xx+yy
    lower = summed < 0.15
    upper = summed >= 0.15 
    primitives[lower, PrimitiveIndex.DENSITY.value] = 0.125
    primitives[lower, PrimitiveIndex.PRESSURE.value] = 0.125
    primitives[upper, PrimitiveIndex.DENSITY.value] = 1.0
    primitives[upper, PrimitiveIndex.PRESSURE.value] = 1.0
    metric = CartesianMinkowski_1_2(grid_info)
    assert(metric.dimension==3) # 1+2 
    return SimulationState(
        primitives,grid_info, bcm, simulation_params, metric
    ) 

class Which2DTestProblem:
    IMPLOSION_TEST=0

def runSim2D(which_sim: Which2DTestProblem):
    match which_sim:
        case Which2DTestProblem.IMPLOSION_TEST:
            state_sim = ImplosionInitialization(t_max=10)            
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
   
if __name__ == "__main__":
#    runSim1D(Which1DTestProblem.CARTESIAN_SOD)
#    Plotting.plot_results_1D()
#    runSim1D(Which1DTestProblem.HARDER_SOD)
#    Plotting.plot_results_1D()
#    runSim1D(Which1DTestProblem.BONDI_PROBLEM)
    # Plotting.plot_Mdot_time("snapshot.pkl")
    # Plotting.plot_Mdot_position("snapshot.pkl")
#    Plotting.plot_results_1D("snapshot.pkl",title="Bondi Accretion", filename="BondiAccretion.png", xlabel="r", show_mach=True, which_slice=10)
#    runSim2D(Which2DTestProblem.IMPLOSION_TEST)
#    Plotting.plot_2D_anim()5098690  