import numpy as np 
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum
from dataclasses import dataclass
import pickle as pkl

class BoundaryCondition(Enum):
    FIXED = 0 
    ZERO_GRAD = 1

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

class TimeUpdate(Enum):
    EULER = 0 
    RK3 = 1

class CoordinateChoice(Enum):
    CARTESIAN = 0 
    SPHERICAL = 1

class WeightType(Enum):
    CENTER = 0 
    EDGE = 1

@dataclass
class SimParams:
    gamma: np.float64
    Courant: np.float64
    t_max: np.float64
    GM: np.float64
    coordinate_system: CoordinateChoice
    include_source: bool
    time_integration: TimeUpdate 

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

    def weights(self, coordinate_system:CoordinateChoice, weight_type: WeightType):
        match coordinate_system:
            case CoordinateChoice.SPHERICAL:
                match weight_type:
                    case WeightType.CENTER:
                        x = self.construct_grid_centers(0) # Assuming the r index is in slot 0 
                    case WeightType.EDGE:
                        x = self.construct_grid_edges(0)
                    case _:
                        raise Exception("Invalid weight type")
                weights  = np.power(x, 2)
            case CoordinateChoice.CARTESIAN:
                match weight_type:
                    case WeightType.CENTER:
                        size = self.NCells[0]
                    case WeightType.EDGE:
                        size = self.NCells[0]+1
                    case _:
                        raise Exception("Invalid weight type")
                weights = np.ones(size)
            case _:
                raise Exception("Invalid coordinate_system")
        return weights
 
    def weight_vector(self, U_cart: npt.ArrayLike, coordinate_system: CoordinateChoice, weight_type: WeightType):
        weights  = self.weights(coordinate_system, weight_type)
        return (weights * U_cart.T).T

    def unweight_vector(self, U: npt.ArrayLike, coordinate_system: CoordinateChoice, weight_type: WeightType):
        weights  = self.weights(coordinate_system, weight_type)
        return (U.T/weights).T

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
    return e

def internal_energy_primitive(W: npt.ArrayLike, simulation_params: SimParams) -> npt.NDArray[np.float64]:
    # primitive variables 
    # W = (\rho, v, P)
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # Invert this: e = P/((\gamma-1) * \rho)
    return W[:,PrimativeIndex.PRESSURE.value]/( (simulation_params.gamma-1) * W[:,PrimativeIndex.DENSITY.value])

def equation_of_state_conservative(U: npt.ArrayLike, global_parameters: InitialParameters) -> npt.NDArray[np.float64]:
    # Return pressure given the conservative variables 
    # EOS assumed (ideal gas): P = (\gamma-1) * \rho * e
    # e is the internal energy
    # gamma is adiabatic index 
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    e = internal_energy_conservative(U)
    assert np.all(e>=0)
    return (global_parameters.simulation_params.gamma-1)*U[:,ConservativeIndex.DENSITY.value]*e

def conservative_to_primitive(U: npt.ArrayLike, global_parameters: InitialParameters) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = U[:,ConservativeIndex.DENSITY.value]
    pressure = equation_of_state_conservative(U, global_parameters)
    assert(np.all(rho!=0))
    return np.stack([rho, U[:, ConservativeIndex.MOMENTUM_DENSITY.value]/ rho, pressure], axis=1) 

def primitive_to_conservative(W: npt.ArrayLike, sim_params: SimParams) -> npt.NDArray[np.float64]:
    # Conservative variables
    #   U = (\rho, \rho*v, E)
    # primitive variables
    #   W = (\rho,v, P)
    rho = W[:,PrimativeIndex.DENSITY.value]
    velocity = W[:,PrimativeIndex.VELOCITY.value]
    # Total energy
    #   E = \rho *e + 0.5*\rho * v**2
    E = rho * internal_energy_primitive(W, sim_params) + 0.5*rho*np.power(velocity,2)
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
    U = U_padded
    primitives = conservative_to_primitive(U, global_parameters)
    c_s = sound_speed(primitives, global_parameters)
    lambda_plus = primitives[:,PrimativeIndex.VELOCITY.value]+c_s
    lambda_minus = primitives[:,PrimativeIndex.VELOCITY.value]-c_s
    # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for + and MAX(0,-left, -right) for - 
    zeros = np.zeros(lambda_plus.shape[0]-2)
    # First, the plus case   
    # Grab the left and right speeds from the padded array
    alpha_plus_candidates = np.stack([zeros, lambda_plus[:-2], lambda_plus[2:]], axis=1)    
    # Calculate max across each row
    alpha_plus = np.max(alpha_plus_candidates, axis=1)
    # Dont forget the minus sign prefactors here! (GPT caught this...) 
    # Prompt: Pasted in source file and stated that internal energy was going negative after 1 timestep; please point out potential errors 
    # It gave me some B.S. about Rusanov flux (which I'm not implementing) and incorrect signage of time update (which is wrong)
    # It did catch the sign error though (which I forgot to add after reading the hydro PDF)
    alpha_minus_candidates = np.stack([zeros, -lambda_minus[:-2], -lambda_minus[2:]], axis=1)    
    alpha_minus = np.max(alpha_minus_candidates, axis=1)
    return (alpha_minus, alpha_plus)

def spatial_derivative(U_padded: npt.ArrayLike, global_parameters: InitialParameters, spatial_index: np.uint = 0) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    # Assuming that U is Cartesian. Cell_scaling is for the fluxes
    cell_flux = flux_from_conservative(U_padded, global_parameters)
    alpha_minus, alpha_plus = alpha_plus_minus(U_padded, global_parameters)
    alpha_sum = alpha_minus+alpha_plus 
    assert(np.all(alpha_sum != 0))
    alpha_prod = alpha_minus*alpha_plus
    # Augment flux array as well with 0 flux conditions 
    # Bunch of .T because numpy broadcasting rules
    left_cell_flux = cell_flux[1:-1,:].T 
    left_conserve = U_padded[1:-1,:].T
    right_cell_flux =cell_flux[2:,:].T  
    right_conserve = U_padded[2:,:].T
    cell_flux_plus_half = (alpha_plus*left_cell_flux+ alpha_minus*right_cell_flux-alpha_prod*(right_conserve-left_conserve))/alpha_sum 
    # Now the minus branch
    left_cell_flux = cell_flux[:-2,:].T 
    left_conserve = U_padded[:-2,:].T
    right_cell_flux =cell_flux[1:-1,:].T  
    right_conserve = U_padded[1:-1,:].T
    cell_flux_minus_half = (alpha_plus*left_cell_flux+ alpha_minus*right_cell_flux-alpha_prod*(right_conserve-left_conserve))/alpha_sum 
    weights = global_parameters.grid_info.weights(global_parameters.simulation_params.coordinate_system, WeightType.EDGE) 
    cell_flux_plus_half_rescaled = cell_flux_plus_half* weights[1:]
    cell_flux_minus_half_rescaled = cell_flux_minus_half* weights[:-1]
    return -(cell_flux_plus_half_rescaled.T-cell_flux_minus_half_rescaled.T)/global_parameters.grid_info.delta()[spatial_index], alpha_plus, alpha_minus

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
    S_1 = 1.0*(2*W[:,PrimativeIndex.PRESSURE.value]*grid_centers-W[:,PrimativeIndex.DENSITY.value]*initial_params.simulation_params.GM)
    S_2 = 1.0*(-W[:,PrimativeIndex.DENSITY.value]*W[:,PrimativeIndex.VELOCITY.value]*initial_params.simulation_params.GM) 
    return np.stack([S_0,S_1,S_2], axis=1) 

def LinearUpdate(U_padded: npt.ArrayLike, global_parameters: InitialParameters): 
    flux_change, alpha_plus, alpha_minus = spatial_derivative(U_padded, global_parameters, 0)
    dt = calc_dt(alpha_plus, alpha_minus, global_parameters)
    state_update = flux_change
    if(global_parameters.simulation_params.include_source):
        state_update += SourceTerm(U_padded[1:-1], global_parameters)
    return dt, state_update

def update(U: npt.ArrayLike, current_time: np.float64, global_parameters: InitialParameters) -> tuple[np.float64, npt.NDArray]:
    assert(U.shape== global_parameters.initial_weighted_U.shape)
    # Undo scaling for input
    U_scaled = global_parameters.grid_info.unweight_vector(U, global_parameters.simulation_params.coordinate_system, WeightType.CENTER)
    U_padded = pad_array(U_scaled, global_parameters )
    dt, state_update_1 = LinearUpdate(U_padded, global_parameters)
    U_1 = U+dt*state_update_1

    match global_parameters.simulation_params.time_integration:
        case TimeUpdate.EULER:
            return current_time+dt, U_1
        case TimeUpdate.RK3:
            U_scaled_1 = global_parameters.grid_info.unweight_vector(U_1, global_parameters.simulation_params.coordinate_system, WeightType.CENTER)
            U_padded_1 = pad_array(U_scaled_1, global_parameters )
            _, state_update_2 = LinearUpdate(U_padded_1, global_parameters)
            U_2 = (3/4)*U+(1/4)*U_1+(1/4)*dt*state_update_2 
            U_scaled_2 = global_parameters.grid_info.unweight_vector(U_2, global_parameters.simulation_params.coordinate_system, WeightType.CENTER)
            U_padded_2 = pad_array(U_scaled_2, global_parameters )
            _, state_update_3 = LinearUpdate(U_padded_2, global_parameters)
            return current_time+dt, (1/3)*U+(2/3)*U_2+(2/3)*dt*state_update_3
        case _:
            raise Exception("Unimplemented TimeUpdate Method")
 
def pad_array(var:npt.ArrayLike,global_parameters: InitialParameters):
    # Augment the array to incorporate the BCs
    # Assuming that var is Cartesian
    # Wastful to do this every time, but it's fine...
    var_initial = global_parameters.grid_info.unweight_vector(global_parameters.initial_weighted_U, global_parameters.simulation_params.coordinate_system, WeightType.CENTER) 
    zero_index_row = var[0,:] 
    last_index_row  = var[-1,:]
    zero_index_row_initial = var_initial[0,:] 
    last_index_row_initial  = var_initial[-1,:]
    left_bc, right_bc = global_parameters.bcm.get_boundary_conds(0) #TODO generalize later
    match right_bc:
        case BoundaryCondition.ZERO_GRAD:
            var_right_pad = np.insert(var, var.shape[0],last_index_row , axis=0)
        case BoundaryCondition.FIXED:
            var_right_pad = np.insert(var, var.shape[0],last_index_row_initial, axis=0)
        case _:
            raise Exception("Unimplemented BC")
    match left_bc:
        case BoundaryCondition.ZERO_GRAD:
            var_padded  = np.insert(var_right_pad, 0, zero_index_row, axis=0) 
        case BoundaryCondition.FIXED:
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
    filename: str = "sod_shock_evolution.png",
    title: str = "Sod Shock Tube Evolution (HLL Flux)",
    xlabel: str = "x",
    show_mach: bool = False
):
# GPT generated w/ edits b/c plotting is a pain...
# Prompt was that after debugging the above, it offered to plot the results 
# I said yes, but use the history list, save it to a png filename
# I also cleaned up extraneous variables and the like
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    N = history[0][1].shape[0]
    support = params.grid_info.construct_grid_centers(0)
    assert(N==support.shape[0])

    if(show_mach):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    t, U = history[-1]
    W = conservative_to_primitive(U, params)

    rho = W[:, PrimativeIndex.DENSITY.value]
    v = W[:, PrimativeIndex.VELOCITY.value]
    P = W[:, PrimativeIndex.PRESSURE.value]
    c_s = sound_speed(W, params)

    label = f"t = {t:.3f}"

    axes[0].plot(support, rho,label=label)
    axes[1].plot(support, v)
    axes[2].plot(support, P)
    if(show_mach):
        axes[3].plot(support, np.abs(v)/c_s )

    axes[0].set_ylabel(r"$\rho$")
    axes[1].set_ylabel(r"$v$")
    axes[2].set_ylabel(r"$P$")
    if(show_mach):
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
                           N_cells: np.int64 = 10000,
                           t_max: np.float64 = 0.2) -> InitialParameters:
    grid_info = GridInfo(np.array([0.0]), np.array([1.0]), np.array([N_cells]))
    simulation_params = SimParams(1.4, 0.5, t_max, 1.0, CoordinateChoice.CARTESIAN,include_source=False, time_integration=TimeUpdate.RK3)
    bcm = BoundaryConditionManager([BoundaryCondition.ZERO_GRAD], [BoundaryCondition.ZERO_GRAD])
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    grid_centers = grid_info.construct_grid_centers(0)
    lower_half = grid_centers<0.5 
    upper_half = grid_centers>=0.5
    primitives[lower_half, PrimativeIndex.DENSITY.value] = rho_l
    primitives[lower_half, PrimativeIndex.VELOCITY.value] = v_l
    primitives[lower_half, PrimativeIndex.PRESSURE.value] = P_l
    primitives[upper_half, PrimativeIndex.DENSITY.value] = rho_r
    primitives[upper_half, PrimativeIndex.VELOCITY.value] = v_r
    primitives[upper_half, PrimativeIndex.PRESSURE.value] = P_r
    initial_conds = primitive_to_conservative(primitives, simulation_params)
    return InitialParameters(grid_info, simulation_params, bcm, initial_conds)

def M_dot(U_profiles: npt.ArrayLike, params: InitialParameters):
    r_center = params.grid_info.construct_grid_centers(0) 
    W = conservative_to_primitive(U_profiles, params)
    return W[:, PrimativeIndex.DENSITY.value]*W[:,PrimativeIndex.VELOCITY.value]*r_center

def BondiAccretionInitialization(
        rho: np.float64,
        v: np.float64,
        P: np.float64,
        N_cells: np.float64
    ):
    grid_info = GridInfo(np.array([0.1]), np.array([1.1]), np.array([N_cells]))
    simulation_params = SimParams(1.4, 0.2, 2.0,1.0, CoordinateChoice.SPHERICAL, include_source=True, time_integration=TimeUpdate.EULER) 
    bcm = BoundaryConditionManager([BoundaryCondition.ZERO_GRAD], [BoundaryCondition.FIXED])
    grid_shape = grid_info.NCells
    primitives = np.zeros( list(grid_shape)+[3]  ) 
    primitives[:, PrimativeIndex.DENSITY.value] = rho
    primitives[:, PrimativeIndex.VELOCITY.value] = v
    primitives[:, PrimativeIndex.PRESSURE.value] = P
    initial_conds = primitive_to_conservative(primitives, simulation_params)
    return InitialParameters(grid_info, simulation_params, bcm, initial_conds)

def CartesianSodProblem():
    initial_params = SodShockInitialization(1.0,0.0,1.0, 0.1, 0.0, 0.125, N_cells=1000, t_max=0.2)
    state = initial_params.initial_weighted_U
    current_time = 0.0 
    history = []
    while(current_time < initial_params.simulation_params.t_max):
        updated_state = update(state, current_time, initial_params)
        history.append(updated_state)
        current_time, state = updated_state
    save_results(history,initial_params)

def HarderSodProblem():
    initial_params = SodShockInitialization(10.0,0.0,100.0, 1.0, 0.0, 1.0, t_max=0.1)
    state = initial_params.initial_weighted_U
    current_time = 0.0 
    history = []
    iteration  = 0
    while(current_time < initial_params.simulation_params.t_max):
        updated_state = update(state, current_time, initial_params)
        if(iteration%100==0):
            #print(iteration, current_time, initial_params.simulation_params.t_max)
            history.append(updated_state)
        history.append(updated_state)
        current_time, state = updated_state
        iteration += 1
    save_results(history,initial_params)

def BondiAccretionProblem():
    initial_params = BondiAccretionInitialization(1.0, 0.0, 0.1, 100)
    state = initial_params.initial_weighted_U 
    current_time = 0.0
    history = []
    iteration = 0
    while(current_time < initial_params.simulation_params.t_max):
        #print(iteration, current_time, initial_params.simulation_params.t_max)
        updated_state = update(state, current_time, initial_params)
        history.append(updated_state)
        current_time, state = updated_state
        iteration += 1
    save_results(history,initial_params) 

def peak_finder(
    input_pkl_file: str = "snapshot.pkl",
    padding:int =100
    ):
    assert(padding%2==0)
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    W = conservative_to_primitive(history[-1][1], params)
    dx = params.grid_info.delta()
    der_prof = (W[1:,:]-W[:-1,:])/(dx) 
    rho_der= der_prof[:, PrimativeIndex.DENSITY.value]
    rho_der_right_pad = np.insert(rho_der, rho_der.shape[0], np.full(padding, -np.inf))
    rho_der_pad = np.insert(rho_der_right_pad, 0, np.full(padding, -np.inf))
    mask = np.full(rho_der_pad.shape, True) 
    for shift in range(-padding,padding+1):
        if(shift==0):
            continue
        shifted_rho_der_pad = np.roll(rho_der_pad, shift)
        new_mask = (rho_der_pad < shifted_rho_der_pad)
        mask &= new_mask 
    unpadded_mask = mask[padding:-padding]
    support = params.grid_info.construct_grid_edges(0)[1:-1]
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 10), sharex=True)

    rho = W[:, PrimativeIndex.DENSITY.value]

    axes.plot(params.grid_info.construct_grid_centers(0), rho)
    #axes.plot(support, rho_der)
    axes.set_ylabel(r"$\rho$")
    axes.legend()
    for pos in support[unpadded_mask]:
        axes.axvline(x=pos, color='black', linestyle='--', linewidth=1)

    plt.show()

def plot_Mdot(
    input_pkl_file: str = "snapshot.pkl"
):
    with open(input_pkl_file, 'rb') as f:
        history, params = pkl.load(f)
    t, profile = history[-1]
    centers=  params.grid_info.construct_grid_centers(0)
    # for time, profile in history:
    #     M_dot_val = M_dot(profile, params)
    #     print(M_dot_val)
    #     t.append(time)
    #     data.append(M_dot_val)
    plt.scatter(centers, M_dot(profile, params))
    plt.show()



if __name__ == "__main__":
    CartesianSodProblem()
    plot_results()
#    peak_finder()
#    HarderSodProblem()
#    plot_results()
#    BondiAccretionProblem()
#    plot_Mdot()
#    plot_results(title="Bondi Accretion", filename="BondiAccretion.png", xlabel="r", show_mach=True)
