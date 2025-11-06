import numpy as np 
import numpy.typing as npt
from enum import Enum
from dataclasses import dataclass
from GridInfo import GridInfo, CoordinateChoice, WeightType
from BoundaryManager import BoundaryConditionManager
from UpdateSteps import TimeUpdateType, SpatialUpdate 

@dataclass
class SimParams:
    gamma: np.float64
    Courant: np.float64
    t_max: np.float64
    GM: np.float64
    coordinate_system: CoordinateChoice
    include_source: bool
    time_integration: TimeUpdateType 
    spatial_integration: SpatialUpdate

class InitialParameters:
    grid_info: GridInfo
    simulation_params: SimParams
    bcm: BoundaryConditionManager
    initial_weighted_U: npt.NDArray # Not necessarily Cartesian

    def __init__(self, a_grid_info, a_simulation_params, a_bcm, a_initial_U):
        self.grid_info = a_grid_info 
        self.simulation_params = a_simulation_params 
        self.bcm = a_bcm 
        # Store the weighted initial conditions
        self.initial_weighted_U = self.grid_info.weight_vector(a_initial_U, self.simulation_params.coordinate_system, WeightType.CENTER)

if __name__=="__main__":
    pass
