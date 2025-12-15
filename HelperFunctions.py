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

## Padding helpers

def initial_value_boundary_padding(vector:npt.NDArray, iaxis_pad_width:tuple[int,int],  iaxis:int , kwargs: dict):
    # Following Note from: https://numpy.org/devdocs/reference/generated/numpy.pad.html
    # vector is a 1D array that is already padded 
    # iaxis_pad_width denotes how many elements of each size are the padded elements  
    #   So the padded portions are vector[:iaxis_pad_width[0]] and vector[-iaxis_pad_width[1]:]
    # iaxis denotes which axis this vector is associated with 
    # kwargs is an empty dictionary
    # Propagate the original edge values to their respective edges
    vector[:iaxis_pad_width[0]] = vector[iaxis_pad_width[0]]
    if(iaxis_pad_width[1]!=0):
        vector[-iaxis_pad_width[1]:] = vector[-iaxis_pad_width[1]-1]

def boundary_padding(vector:npt.NDArray, iaxis_pad_width:tuple[int,int],  iaxis:int , 
                     initial_boundary: npt.NDArray, left_bc: BoundaryCondition, right_bc: BoundaryCondition):
    # Following Note from: https://numpy.org/devdocs/reference/generated/numpy.pad.html
    # vector is a 1D array that is already padded 
    # iaxis_pad_width denotes how many elements of each size are the padded elements  
    #   So the padded portions are vector[:iaxis_pad_width[0]] and vector[-iaxis_pad_width[1]:]
    # iaxis denotes which axis this vector is associated with 
    # kwargs is an empty dictionary
    match left_bc:
        case BoundaryCondition.ZERO_GRAD:
            vector[:iaxis_pad_width[0]] = vector[iaxis_pad_width[0]]
        case BoundaryCondition.FIXED:
            vector[:iaxis_pad_width[0]] = initial_boundary[iaxis_pad_width[0]]
        case _:
            raise Exception("Unimplemented BC")
    match right_bc:
        case BoundaryCondition.ZERO_GRAD:
            if(iaxis_pad_width[1]!=0):
                vector[-iaxis_pad_width[1]:] = vector[-iaxis_pad_width[1]-1]
        case BoundaryCondition.FIXED:
            if(iaxis_pad_width[1]!=0):  
                vector[-iaxis_pad_width[1]:] = initial_boundary[-iaxis_pad_width[1]-1]
        case _:
            raise Exception("Unimplemented BC")

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