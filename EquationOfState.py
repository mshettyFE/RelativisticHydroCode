from CommonClasses import *
from CommonClasses import SimParams

def equation_of_state_primitive(sim_params: SimParams, pressure: npt.ArrayLike, density: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Calculates internal energy as a function of pressure and density
    return pressure /( (sim_params.gamma-1) * density) 

def equation_of_state_epsilon(sim_params: SimParams, epsilon: npt.ArrayLike, density: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Calculates pressure as a function of internal energy (epsilon) and density
    return  (sim_params.gamma-1) * density * epsilon

def sound_speed(simulation_params: SimParams, pressure: npt.ArrayLike, density: npt.ArrayLike, which_regime: WhichRegime = WhichRegime.NEWTONIAN) -> npt.NDArray[np.float64]:
    # Return sound speed given pressure and density
    match which_regime:
        case WhichRegime.RELATIVITY:
            h = internal_enthalpy_primitive_raws(pressure,density,simulation_params)
            return np.sqrt(simulation_params.gamma* pressure / density/h)
        case WhichRegime.NEWTONIAN:
            return np.sqrt(simulation_params.gamma* pressure / density)
    return 

def internal_enthalpy_primitive_raws(pressure, density, sim_params: SimParams) -> npt.NDArray[np.float64]:
    internal_energy  =equation_of_state_primitive(sim_params, pressure, density )
    return 1 + internal_energy + pressure/density
