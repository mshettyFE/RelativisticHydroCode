import numpy as np 
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
import pickle as pkl

class BoundaryCondition(Enum):
    INFLOW = 0 
    OUTFLOW = 1

class BoundaryConditionManager:
    left_bcs: list[BoundaryCondition]
    right_bcs: list[BoundaryCondition]

    def __init__(self, left_boundaries: list[BoundaryCondition], right_boundaries: list[BoundaryCondition]):
        assert(len(left_boundaries) == len(right_boundaries))
        for left, right in zip(left_boundaries, right_boundaries):
            assert(isinstance(left, BoundaryCondition))
            assert(isinstance(right, BoundaryCondition))
        self.left_bcs = left_boundaries 
        self.right_bcs = right_boundaries 

    def get_boundary_conds(self, index: int):
        assert( (index>=0) and (index < len(self.left_bcs)))
        return (self.left_bcs[index], self.right_bcs[index])

@dataclass
class SimParams:
    gamma: np.float64
    Courant: np.float64
    t_max: np.float64
    GM: np.float64

class GridInfo:
    leftmost_edges: npt.NDArray
    rightmost_edges: npt.NDArray
    NCells: npt.NDArray

    def __init__(self, left: npt.ArrayLike, right: npt.ArrayLike, cell_count: npt.ArrayLike):
        assert(left.shape==right.shape)
        assert(right.shape==cell_count.shape)
        assert(left.ndim==1)
        assert(right.ndim==1)
        assert(cell_count.ndim==1)
        self.leftmost_edges = left
        self.rightmost_edges = right
        self.NCells = cell_count

    def delta(self):
        return 1.0/(self.NCells+1.0) 

    def construct_grid_edges(self,index: np.uint):
        # Need +1  in order to generate NCells when using construct_grid_centers()
        return np.linspace(self.leftmost_edges[index], self.rightmost_edges[index], self.NCells[index]+1 )
    
    def construct_grid_centers(self, index: np.uint):
        grid_edges = self.construct_grid_edges(index)
        return 0.5*(grid_edges[1:]+grid_edges[:-1])


@dataclass
class InitialParameters:
    grid_info: GridInfo
    simulation_params: SimParams
    initial_U: npt.NDArray   # NB I'm assuming that the dim(U) is Nx3
    bcm: BoundaryConditionManager

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

def internal_energy_primitive(W: npt.ArrayLike, global_parameters: InitialParameters) -> npt.NDArray[np.float64]:
    # primitive variables 
    # W = (\rho, v, P)
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # Invert this: e = P/((\gamma-1) * \rho)
    return W[:,PrimativeIndex.PRESSURE.value]/( (global_parameters.simulation_params.gamma-1) * W[:,PrimativeIndex.DENSITY.value])

def equation_of_state_conservative(U: npt.ArrayLike, global_parameters: InitialParameters) -> npt.NDArray[np.float64]:
    # Return pressure given the conservative variables 
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # e is the internal energy
    # gamma is adiabatic index 
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    return (global_parameters.simulation_params.gamma-1)*U[:,ConservativeIndex.DENSITY.value]*internal_energy_conservative(U)

def conservative_to_primitive(U: npt.ArrayLike, global_parameters: InitialParameters) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = U[:,ConservativeIndex.DENSITY.value]
    pressure = equation_of_state_conservative(U, global_parameters)
    assert(np.all(rho!=0))
    return np.stack([rho, U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/ rho, pressure], axis=1) 

def primitive_to_conservative(W: npt.ArrayLike, global_parameters: InitialParameters) -> npt.NDArray[np.float64]:
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

def flux_from_conservative(U: npt.ArrayLike, global_parameters: InitialParameters) -> npt.NDArray[np.float64]:
    primitive = conservative_to_primitive(U,global_parameters)
    F_0 = U[:,ConservativeIndex.MOMENTUM_DENSITY.value]
    F_1 = F_0*primitive[:,PrimativeIndex.VELOCITY.value]+primitive[:,PrimativeIndex.PRESSURE.value]
    F_2 = (U[:,ConservativeIndex.ENERGY.value]+primitive[:,PrimativeIndex.PRESSURE.value])*primitive[:,PrimativeIndex.VELOCITY.value]
    return np.stack([F_0,F_1,F_2], axis=1)

def sound_speed(W: npt.ArrayLike, global_parameters: InitialParameters):
    return np.sqrt(global_parameters.simulation_params.gamma*W[:, PrimativeIndex.PRESSURE.value]/ W[:, PrimativeIndex.DENSITY.value])

def alpha_plus_minus(U_padded: npt.ArrayLike, global_parameters: InitialParameters) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    U = U_padded[1:-1]
    primitives = conservative_to_primitive(U, global_parameters)
    c_s = sound_speed(primitives, global_parameters)
    lambda_plus = primitives[:,PrimativeIndex.VELOCITY.value]+c_s
    lambda_minus = primitives[:,PrimativeIndex.VELOCITY.value]-c_s
    # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for + and MAX(0,-left, -right) for - 
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
    # It did catch the sign error though (which I forgot to add after reading the hydro PDF)
    alpha_minus_candidates = np.stack([zeros, -lambda_minus_padded[:-2], -lambda_minus_padded[2:]], axis=1)    
    alpha_minus = np.max(alpha_minus_candidates, axis=1)
    return (alpha_minus, alpha_plus)

def spatial_derivative(U_padded: npt.ArrayLike, global_parameters: InitialParameters, spatial_index: np.uint = 0, scale_cell: bool = False) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # Assuming that U is Cartesian. Cell_scaling is for the fluxes
    cell_flux = flux_from_conservative(U_padded, global_parameters)
    alpha_minus, alpha_plus = alpha_plus_minus(U_padded, global_parameters)
    alpha_sum = alpha_minus+alpha_plus 
    assert(np.all(alpha_sum != 0))
    alpha_prod = alpha_minus*alpha_plus
    # Augment flux array as well with 0 flux conditions 
    # Bunch of .T because numpy broadcasting rules
    cell_flux_plus_half = (alpha_plus*cell_flux[1:-1,:].T+ alpha_minus*cell_flux[2:,:].T-alpha_prod*(U_padded[2:,:].T-U_padded[1:-1,:].T))/alpha_sum 
    cell_flux_minus_half = (alpha_plus*cell_flux[:-2,:].T+alpha_minus*cell_flux[1:-1,:].T-alpha_prod*(U_padded[1:-1,:].T-U_padded[2:,:].T))/alpha_sum
    if(scale_cell):
        grid_edges = global_parameters.grid_info.construct_grid_edges(spatial_index)
        cell_flux_plus_half_rescaled = cell_flux_plus_half* np.power(grid_edges[1:],2)
        cell_flux_minus_half_rescaled = cell_flux_minus_half* np.power(grid_edges[:-1],2)
        return -(cell_flux_plus_half_rescaled.T-cell_flux_minus_half_rescaled.T)/global_parameters.grid_info.delta()[spatial_index], alpha_plus, alpha_minus
    else:
        return -(cell_flux_plus_half.T-cell_flux_minus_half.T)/global_parameters.grid_info.delta()[spatial_index], alpha_plus, alpha_minus

def calc_dt(alpha_plus: npt.ArrayLike, alpha_minus:npt.ArrayLike, global_parameters: InitialParameters):
    max_alpha = np.max( [alpha_plus, alpha_minus]) 
    return global_parameters.simulation_params.Courant*np.min(global_parameters.grid_info.delta())/max_alpha

def SourceTerm(U:npt.ArrayLike, initial_params: InitialParameters):
    # Assumes that U is the array of Cartesian conservative variables
    W = conservative_to_primitive(U, initial_params)
    grid_centers = initial_params.grid_info.construct_grid_centers(0)
    # primitive variables 
    # W = (\rho, v, P)
    S_0 = np.zeros(grid_centers.shape)
    S_1 = 2*W[:,PrimativeIndex.PRESSURE.value]*grid_centers-W[:,PrimativeIndex.DENSITY.value]*initial_params.simulation_params.GM
    S_2 = -W[:,PrimativeIndex.DENSITY.value]*W[:,PrimativeIndex.VELOCITY.value]*initial_params.simulation_params.GM 
    return np.stack([S_0,S_1,S_2], axis=1) 

def update(U: npt.ArrayLike, current_time: np.float64, global_parameters: InitialParameters, scale_cells:bool =False) -> tuple[np.float64, npt.NDArray]:
    assert(U.shape== global_parameters.initial_U.shape)
    if(scale_cells==False):
        U_scaled = U
    else:
        # Here, we are assuming that U is of the form r^{2} U_{cart}
        # .T because broadcasting rules 
        U_scaled = (U.T/spherical_weights(U, global_parameters)).T
    U_padded = pad_array(U_scaled, global_parameters )
    flux_change, alpha_plus, alpha_minus = spatial_derivative(U_padded, global_parameters, 0, scale_cells)
    dt = calc_dt(alpha_plus, alpha_minus, global_parameters)
    if(scale_cells==False):
        state_update = flux_change*dt
    else:
        state_update = dt*(flux_change+SourceTerm(U_scaled, global_parameters))
    return current_time+dt, U+state_update

def spherical_weights(U: npt.ArrayLike, global_params: InitialParameters):
    r_grid = global_params.grid_info.construct_grid_centers(0) # Assuming the r index is in slot 0 
    assert(r_grid.shape[-1]==U.T.shape[-1])
    return np.power(r_grid,2)

def pad_array(var:npt.ArrayLike,global_parameters: InitialParameters):
    # Augment the array to incorporate the BCs. defaults to 0 flux at the boundaries (ie. copy the edge values to the ghost cells)
    var_initial = global_parameters.initial_U
    zero_index_row = var[0,:] 
    last_index_row  = var[-1,:]
    zero_index_row_initial = var_initial[0,:] 
    last_index_row_initial  = var_initial[-1,:]
    left_bc, right_bc = global_parameters.bcm.get_boundary_conds(0) #TODO generalize later
    match right_bc:
        case BoundaryCondition.OUTFLOW:
            var_right_pad = np.insert(var, var.shape[0],last_index_row , axis=0)
        case BoundaryCondition.INFLOW:
            var_right_pad = np.insert(var, var.shape[0],last_index_row_initial, axis=0)
        case _:
            raise Exception("Unimplemented BC")
    match left_bc:
        case BoundaryCondition.OUTFLOW:
            var_padded  = np.insert(var_right_pad, 0, zero_index_row, axis=0) 
        case BoundaryCondition.INFLOW:
            var_padded  = np.insert(var_right_pad, 0, zero_index_row_initial, axis=0) 
        case _:
            raise Exception("Unimplemented BC")
    return var_padded    

def save_results(
        history: list[tuple[np.float64, npt.NDArray]],
        params: InitialParameters,
        filename: str = "snapshot.pkl"
        ):
    data = (history, params)
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

def plot_results(
    input_pkl_file: str = "snapshot.pkl",
    n_snapshots: int = 6,
    filename: str = "sod_shock_evolution.png",
    title: str = "Sod Shock Tube Evolution (HLL Flux)",
    xlabel: str = "x"
):
# GPT generated w/ edits b/c plotting is a pain...
# Prompt was that after debugging the above, it offered to plot the results 
# I said yes, but use the history list, save it to a png file, and only plot every n_snapshot profiles
# I also cleaned up extraneous variables and the like
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    N = history[0][1].shape[0]
    support = params.grid_info.construct_grid_centers(0)
    assert(N==support.shape[0])

    # Choose evenly spaced snapshots
    indices = np.linspace(0, len(history) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    for i, idx in enumerate(indices):
        t, U = history[idx]
        W = conservative_to_primitive(U, params)

        rho = W[:, PrimativeIndex.DENSITY.value]
        v = W[:, PrimativeIndex.VELOCITY.value]
        P = W[:, PrimativeIndex.PRESSURE.value]
        c_s = sound_speed(W, params)

        color = plt.cm.viridis(i / (n_snapshots - 1))
        label = f"t = {t:.3f}"

        axes[0].plot(support, rho, color=color, label=label)
        axes[1].plot(support, v, color=color)
        axes[2].plot(support, P, color=color)
        axes[3].plot(support, np.abs(v)/c_s, color=color)

    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$v$")
    axes[2].set_ylabel(r"$P$")
    axes[3].set_ylabel(r"$M$")
    axes[3].axhline(y=1, color='black', linestyle='--', linewidth=1)
    for i, idx in enumerate(axes):
        axes[i].set_xlabel(xlabel)

    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(title, fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200)

def SodShockInitialization(rho_l: np.float64, v_l: np.float64, P_l: np.float64,
                           rho_r:np.float64, v_r: np.float64, P_r:np.float64,
                           N_cells: np.int64 = 1000) -> InitialParameters:
    grid_info = GridInfo(np.array([0.0]), np.array([1.0]), np.array([N_cells]))
    simulation_params = SimParams(1.4, 0.5, 0.2, 1.0)
    bcm = BoundaryConditionManager([BoundaryCondition.OUTFLOW], [BoundaryCondition.OUTFLOW])
    Params = InitialParameters(grid_info, simulation_params, None, bcm)
    grid_shape = Params.grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    grid_centers = Params.grid_info.construct_grid_centers(0)
    lower_half = grid_centers<0.5 
    upper_half = grid_centers>=0.5
    primitives[lower_half, PrimativeIndex.DENSITY.value] = rho_l
    primitives[lower_half, PrimativeIndex.VELOCITY.value] = v_l
    primitives[lower_half, PrimativeIndex.PRESSURE.value] = P_l
    primitives[upper_half, PrimativeIndex.DENSITY.value] = rho_r
    primitives[upper_half, PrimativeIndex.VELOCITY.value] = v_r
    primitives[upper_half, PrimativeIndex.PRESSURE.value] = P_r
    initial_conds = primitive_to_conservative(primitives, Params)
    Params.initial_U = initial_conds
    return Params

def BondiAccretionInitialization(
        rho: np.float64,
        v: np.float64,
        P: np.float64,
        N_cells: np.float64
    ):
    grid_info = GridInfo(np.array([0.1]), np.array([1.1]), np.array([N_cells]))
    simulation_params = SimParams(1.4, 0.5, 2.0,1.0) 
    bcm = BoundaryConditionManager([BoundaryCondition.OUTFLOW], [BoundaryCondition.INFLOW])
    Params = InitialParameters(grid_info, simulation_params, None, bcm)
    grid_shape = Params.grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    primitives[:, PrimativeIndex.DENSITY.value] = rho
    primitives[:, PrimativeIndex.VELOCITY.value] = rho
    primitives[:, PrimativeIndex.PRESSURE.value] = rho
    initial_conds = primitive_to_conservative(primitives, Params)
    Params.initial_U = initial_conds
    return Params

def CartesianSodProblem():
    initial_params = SodShockInitialization(1.0,0.0,1.0, 0.125, 0.0, 0.1)
    state = initial_params.initial_U
    current_time = 0.0 
    history = []
    while(current_time < initial_params.simulation_params.t_max):
        updated_state = update(state, current_time, initial_params)
        history.append(updated_state)
        current_time, state = updated_state
    save_results(history,initial_params)

def BondiAccretionProblem():
    initial_params = BondiAccretionInitialization(1.0, 0.0, 0.1, 100)
    state = initial_params.initial_U 
    current_time = 0.0
    history = []
    while(current_time < initial_params.simulation_params.t_max):
        updated_state = update(state, current_time, initial_params, scale_cells=True)
        history.append(updated_state)
        current_time, state = updated_state
    save_results(history,initial_params) 

if __name__ == "__main__":
#    CartesianSodProblem()
#    plot_results()
    BondiAccretionProblem()
    plot_results(title="Bondi Accretion", xlabel="r")
