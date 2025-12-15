from HydroCore import SimParams
import numpy.typing as npt
import numpy as np
from metrics import Metric
from HelperFunctions import index_conservative_var, index_primitive_var, PrimitiveIndex, ConservativeIndex
from EquationOfState import *

def pressure_finding_function(guess_pressure: npt.ArrayLike,
                              U_cart: npt.ArrayLike, 
                               metric: Metric ) -> npt.NDArray:
    # Implements p(\rho(\bar{p}))
    pass

def pressure_finding_func_der(guess_pressure, 
                               U_cart: npt.ArrayLike, 
                               metric: Metric ) -> npt.NDArray:
    # Implements f'(p) = |v(p)|^{2} c_{s}^{2}-1
    pass

def velocity_guess(guess_pressure: npt.ArrayLike, U_cart: npt.ArrayLike) -> npt.NDArray:
    # v^{i} = S^{i}/(D+\tau+guess_p)
    pass

def internal_energy_guess(guess_pressure: npt.ArrayLike, U_cart: npt.ArrayLike) -> npt.NDArray:
    # (\tau+D(1-W)+p(1-W)^{2})
    pass

def density_guess(guess_pressure: npt.ArrayLike, U_cart: npt.ArrayLike ) -> npt.NDArray:
    #\rho  = D/W
    pass

def construct_primitives_from_guess(guess_pressure:npt.ArrayLike, U_cart: npt.ArrayLike, metric: Metric) -> npt.NDArray:
    pass