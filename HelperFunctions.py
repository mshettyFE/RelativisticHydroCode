import numpy.typing as npt
from enum import Enum
import numpy as np
from dataclasses import dataclass
from UpdateSteps import TimeUpdateType,SpatialUpdate
from BoundaryManager import BoundaryCondition 

class WhichRegime(Enum):
    NEWTONIAN = 0
    RELATIVITY = 1

class WhichVar(Enum):
    PRIMITIVE  =0
    CONSERVATIVE   =1


# NOTE: Is there a confusion between internal energy and internal enthalpy usage?
class PrimitiveIndex(Enum):
    DENSITY = 0
    PRESSURE = 1
    X_VELOCITY = 2
    Y_VELOCITY = 3
    Z_VELOCITY = 4

class ConservativeIndex(Enum):
    DENSITY = 0
    TAU = 1
    X_MOMENTUM_DENSITY =2
    Y_MOMENTUM_DENSITY = 3
    Z_MOMENTUM_DENSITY = 4

@dataclass
class SimParams:
    gamma: np.float64
    Courant: np.float64
    t_max: np.float64
    GM: np.float64
    include_source: bool
    time_integration: TimeUpdateType 
    spatial_integration: SpatialUpdate
    regime: WhichRegime 

## Indexing variable tensors

def index_conservative_var(U_cart: npt.ArrayLike, var_type: ConservativeIndex, n_variable_dims: int): 
    match n_variable_dims:
        case 1:
            max_allowed_index = ConservativeIndex.X_MOMENTUM_DENSITY
        case 2:
            max_allowed_index = ConservativeIndex.Y_MOMENTUM_DENSITY 
        case 3:
            max_allowed_index = ConservativeIndex.Z_MOMENTUM_DENSITY 
        case _:
            raise Exception("Unimplemented simulation dimension") 
    if(var_type.value <= max_allowed_index.value):
        return U_cart[...,var_type.value]
    raise Exception("Trying to index momentum that is larger than the dimension of the problem") 

def index_primitive_var( primitive: npt.NDArray, var_type: PrimitiveIndex, n_variable_dims: int):
    match n_variable_dims:
        case 1:
            max_allowed_index = PrimitiveIndex.X_VELOCITY
        case 2:
            max_allowed_index = PrimitiveIndex.Y_VELOCITY
        case 3:
            max_allowed_index = PrimitiveIndex.Z_VELOCITY
        case _:
            raise Exception("Unimplemented simulation dimension")
    if(var_type.value <= max_allowed_index.value):
        return primitive[..., var_type.value] 
    raise Exception("Trying to index momentum that is larger than the dimension of the problem")