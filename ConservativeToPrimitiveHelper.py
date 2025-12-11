from HydroCore import SimulationState, ConservativeIndex, PrimitiveIndex
import numpy.typing as npt
import numpy as np
from GridInfo import WeightType

def primitive_from_pressure(pressure: npt.NDArray, U_cart: npt.NDArray) -> npt.NDArray:
    pass

def pressure_finding_function(guess_pressure, sim_state: SimulationState) -> npt.NDArray:
    pass

def pressure_finding_func_der(guess_pressure, sim_state: SimulationState) -> npt.NDArray:
    pass