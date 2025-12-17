from HelperFunctions import *
from HelperFunctions import SimParams

def equation_of_state_primitive(sim_params: SimParams, pressure: npt.ArrayLike, density: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Calculates internal energy as a function of pressure and density
    return pressure /( (sim_params.gamma-1) * density) 

def pressure_from_epsilon(sim_params: SimParams, epsilon: npt.ArrayLike, density: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Calculates pressure as a function of internal energy (epsilon) and density
    return  (sim_params.gamma-1) * density * epsilon

def sound_speed(simulation_params: SimParams, pressure: npt.ArrayLike, density: npt.ArrayLike):
    # Return sound speed given pressure and density
    return np.sqrt(simulation_params.gamma* pressure / density)

def internal_enthalpy_primitive(W: npt.ArrayLike, sim_params: SimParams,  n_variable_dims) -> npt.NDArray[np.float64]:
    pressure = index_primitive_var(W, PrimitiveIndex.PRESSURE,n_variable_dims)
    density = index_primitive_var(W, PrimitiveIndex.DENSITY,n_variable_dims)
    internal_energy  =equation_of_state_primitive(sim_params, pressure, density )
    return 1 + internal_energy + pressure/density

def internal_enthalpy_from_internal_energy(epsilon: npt.ArrayLike, sim_params: SimParams) -> npt.NDArray[np.float64]:
    return 1+sim_params.gamma*epsilon