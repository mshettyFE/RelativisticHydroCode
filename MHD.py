import numpy as np 
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import TypedDict
from enum import Enum

counter = 0
PARAM_DICT_TYPING =  TypedDict("Params", {"gamma": np.float64, "dx": np.float64, "Courant": 0.5})
#  Params = {"gamma": 2.0, "dx": 0.05, "Courant":0.5}

# NB I'm assuming that the dim(U) is Nx3

class PrimativeIndex(Enum):
    DENSITY = 0
    VELOCITY = 1
    PRESSURE = 2

class ConservativeIndex(Enum):
    DENSITY = 0
    MOMENTUM_DENSITY = 1
    ENERGY = 2


def internal_energy_conservative(U: npt.ArrayLike) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    rho = U[:,ConservativeIndex.DENSITY.value]
    assert(np.all(rho!=0))
    v = U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/rho
    E = U[:,ConservativeIndex.ENERGY.value]
    # Total energy
    #   E = \rho *e + 0.5*\rho * v**2
    # Inverting 
    #  E/rho-0.5 v**2 = e
    e = E/rho-0.5*np.power(v,2)
    assert np.all(e>=0)
    return e

def internal_energy_primitive(W: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # primitive variables 
    # W = (\rho, v, P)
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # Invert this: e = P/((\gamma-1) * \rho)
    return W[:,PrimativeIndex.PRESSURE.value]/( (global_parameters["gamma"]-1) * W[:,PrimativeIndex.DENSITY.value])

def equation_of_state_conservative(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # Return pressure given the conservative variables 
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # e is the internal energy
    # gamma is adiabatic index 
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    return (global_parameters["gamma"]-1)*U[:,ConservativeIndex.DENSITY.value]*internal_energy_conservative(U)

def conservative_to_primitive(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = U[:,ConservativeIndex.DENSITY.value]
    pressure = equation_of_state_conservative(U, global_parameters)
    assert(np.all(rho!=0))
    return np.stack([rho, U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/ rho, pressure], axis=1) 

def primitive_to_conservative(W: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = W[:,PrimativeIndex.DENSITY.value]
    velocity = W[:,PrimativeIndex.VELOCITY.value]
    # Total energy
    #   E = \rho *e + 0.5*\rho * v**2
    E = rho * internal_energy_primitive(W, global_parameters) + 0.5*rho*np.power(velocity,2)
    return np.stack([rho, rho*velocity, E], axis= 1) 

def flux_from_conservative(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> npt.NDArray[np.float64]:
    primitive = conservative_to_primitive(U,global_parameters)
    F_0 = U[:,ConservativeIndex.MOMENTUM_DENSITY.value]
    F_1 = F_0*primitive[:,PrimativeIndex.VELOCITY.value]+primitive[:,PrimativeIndex.PRESSURE.value]
    F_2 = (U[:,ConservativeIndex.ENERGY.value]+primitive[:,PrimativeIndex.PRESSURE.value])*primitive[:,PrimativeIndex.VELOCITY.value]
    return np.stack([F_0,F_1,F_2], axis=1)

def alpha_plus_minus(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    primitives = conservative_to_primitive(U, global_parameters)
    c_s = np.sqrt(global_parameters["gamma"]*primitives[:, PrimativeIndex.PRESSURE.value]/ primitives[:, PrimativeIndex.DENSITY.value])
    lambda_plus = primitives[:,PrimativeIndex.VELOCITY.value]+c_s
    lambda_minus = primitives[:,PrimativeIndex.VELOCITY.value]-c_s
    # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for both + and - 
    zeros = np.zeros(lambda_plus.shape[0])
    # First, the plus case   
    # Augment the eigenvalue array with 0 at the start and end
    lambda_plus_right = np.insert(lambda_plus, lambda_plus.shape[0], 0)
    lambda_plus_padded = np.insert(lambda_plus_right, 0, 0) 
    # Grab the left and right speeds from the padded array
    alpha_plus_candidates = np.stack([zeros, lambda_plus_padded[:-2], lambda_plus_padded[2:]], axis=1)    
    # Calculate max across each row
    alpha_plus = np.max(alpha_plus_candidates, axis=1)
    # minus case
    lambda_minus_right = np.insert(lambda_minus, lambda_minus.shape[0], 0)
    lambda_minus_padded = np.insert(lambda_minus_right, 0, 0) 
    # Dont forget the minus sign prefactors here! (GPT caught this...) 
    # Prompt: Pasted in source file and stated that internal energy was going negative after 1 timestep; please point out potential errors 
    # It gave me some B.S. about Rusanov flux (which I'm not implementing) and incorrect signage of time update (which is wrong)
    # It did catch tiis sign error though (which I forgot to add after reading the hydro PDF)
    alpha_minus_candidates = np.stack([zeros, -lambda_minus_padded[:-2], -lambda_minus_padded[2:]], axis=1)    
    alpha_minus = np.max(alpha_minus_candidates, axis=1)
    return (alpha_minus, alpha_plus)

def spatial_derivative(U: npt.ArrayLike, global_parameters: PARAM_DICT_TYPING) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # Augment the array to have 0 flux at the boundaries (ie. copy the edge values to the ghost cells)
    zero_index_row = U[0,:] 
    last_index_row  = U[-1,:]
    U_right_pad = np.insert(U, U.shape[0],last_index_row , axis=0)
    U_padded  = np.insert(U_right_pad, 0, zero_index_row, axis=0) 
    cell_flux = flux_from_conservative(U_padded, global_parameters)
    alpha_minus, alpha_plus = alpha_plus_minus(U, global_parameters)
    alpha_sum = alpha_minus+alpha_plus 
    assert(np.all(alpha_sum != 0))
    alpha_prod = alpha_minus*alpha_plus
    # Augment flux array as well with 0 flux conditions 
    # Bunch of .T because numpy broadcasting rules
    cell_flux_plus_half = (alpha_plus*cell_flux[1:-1,:].T+ alpha_minus*cell_flux[2:,:].T-alpha_prod*(U_padded[2:,:].T-U_padded[1:-1,:].T))/alpha_sum 
    cell_flux_minus_half = (alpha_plus*cell_flux[:-2,:].T+alpha_minus*cell_flux[1:-1,:].T-alpha_prod*(U_padded[1:-1,:].T-U_padded[2:,:].T))/alpha_sum
    return -(cell_flux_plus_half.T-cell_flux_minus_half.T)/global_parameters["dx"], alpha_plus, alpha_minus

def calc_dt(alpha_plus: npt.ArrayLike, alpha_minus:npt.ArrayLike, global_parameters: PARAM_DICT_TYPING):
    max_alpha = np.max( [alpha_plus, alpha_minus]) 
    return global_parameters["Courant"]*global_parameters["dx"]/max_alpha

def update(U: npt.ArrayLike, current_time: np.float64, global_parameters: PARAM_DICT_TYPING) -> tuple[np.float64, npt.NDArray]:
    flux_change, alpha_plus, alpha_minus = spatial_derivative(U, global_parameters)
    dt = calc_dt(alpha_plus, alpha_minus, global_parameters)
    state_update = flux_change*dt
    return current_time+dt, U+state_update

def plot_results(
    history: list[tuple[np.float64, npt.NDArray]],
    params: PARAM_DICT_TYPING,
    n_snapshots: int = 6,
    filename: str = "sod_shock_evolution.png"
):
    # GPT generated w/ edits b/c plotting is a pain...
    # Prompt was that after debugging the above, it offered to plot the results 
    # I said yes, but use the history list, save it to a png file, and only plot every n_snapshot profiles
    # I also cleaned up extraneous variables and the like
    """
    Plot and save density, velocity, and pressure profiles from the simulation history.

    Parameters
    ----------
    history : list of (time, state)
        List of tuples from the time integration loop.
    params : dict
        Global simulation parameters (contains gamma, dx, etc.).
    n_snapshots : int
        Number of evenly spaced snapshots to plot.
    filename : str
        Output filename for saved PNG figure.
    """
    N = history[0][1].shape[0]
    x = np.linspace(0.0, 1.0, N)

    # Choose evenly spaced snapshots
    indices = np.linspace(0, len(history) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for i, idx in enumerate(indices):
        t, U = history[idx]
        W = conservative_to_primitive(U, params)

        rho = W[:, PrimativeIndex.DENSITY.value]
        v = W[:, PrimativeIndex.VELOCITY.value]
        P = W[:, PrimativeIndex.PRESSURE.value]

        color = plt.cm.viridis(i / (n_snapshots - 1))
        label = f"t = {t:.3f}"

        axes[0].plot(x, rho, color=color, label=label)
        axes[1].plot(x, v, color=color)
        axes[2].plot(x, P, color=color)

    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$v$")
    axes[2].set_ylabel(r"$P$")
    for i, idx in enumerate(axes):
        axes[i].set_xlabel(r"x")

    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("Sod Shock Tube Evolution (HLL Flux)", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200)

def SodShockInitialization(rho_l: np.float64, v_l: np.float64, P_l: np.float64,
                           rho_r:np.float64, v_r: np.float64, P_r:np.float64,
                           N: np.int64 = 1000) -> tuple[npt.NDArray, PARAM_DICT_TYPING]:
    Params = {"gamma": 1.4, "dx": 1/N, "Courant":0.5}
    primitives = np.zeros( (N,3)) 
    grid = np.linspace(0.0,1.0,N)
    lower_half = grid<0.5 
    upper_half = grid>=0.5
    primitives[lower_half, PrimativeIndex.DENSITY.value] = rho_l
    primitives[lower_half, PrimativeIndex.VELOCITY.value] = v_l
    primitives[lower_half, PrimativeIndex.PRESSURE.value] = P_l
    primitives[upper_half, PrimativeIndex.DENSITY.value] = rho_r
    primitives[upper_half, PrimativeIndex.VELOCITY.value] = v_r
    primitives[upper_half, PrimativeIndex.PRESSURE.value] = P_r
    return primitive_to_conservative(primitives, Params), Params

if __name__ == "__main__":
    t_max = 0.2
    initial_cond, param_dict = SodShockInitialization(1.0,0.0,1.0, 0.125, 0.0, 0.1)
    state, params = SodShockInitialization(1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    current_time = 0.0 
    history = []
    while(current_time < t_max):
        updated_state = update(state, current_time, params)
        history.append(updated_state)
        current_time, state = updated_state
plot_results(history, params, n_snapshots=6, filename="sod_shock_evolution.png")
