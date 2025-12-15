from HelperFunctions import *

def equation_of_state_primitive(sim_params: SimParams, pressure: npt.ArrayLike, density: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Calculates internal energy as a function of pressure and density
    return pressure /( (sim_params.gamma-1) * density) 

def pressure_from_epsilon(sim_params: SimParams, epsilon: npt.ArrayLike, density: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Calculates pressure as a function of internal energy (epsilon) and density
    return  (sim_params.gamma-1) * density * epsilon

def sound_speed(simulation_params: SimParams, pressure: npt.ArrayLike, density: npt.ArrayLike):
    # Return sound speed given pressure and density
    return np.sqrt(simulation_params.gamma* pressure / density)