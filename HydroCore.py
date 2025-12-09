import numpy.typing as npt
from enum import Enum
import numpy as np
from numpy.lib._index_tricks_impl import ndindex
from numpy.lib._arraypad_impl import _as_pairs
from dataclasses import dataclass
from BoundaryManager import BoundaryCondition 
from GridInfo import GridInfo, WeightType
from UpdateSteps import TimeUpdateType,SpatialUpdate, SpatialUpdateType
from BoundaryManager import BoundaryConditionManager
from metrics import Metric
from metrics.Metric import WhichCacheTensor, METRIC_VARIABLE_INDEX

class PrimitiveIndex(Enum):
    DENSITY = 0
    PRESSURE = 1
    X_VELOCITY = 2
    Y_VELOCITY = 3
    Z_VELOCITY = 4

class ConservativeIndex(Enum):
    DENSITY = 0
    ENERGY = 1
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


class SimulationState:
    U: npt.ArrayLike # Conservative Variables 
    U_initial_unweighted_padded: npt.ArrayLike # Initial conditions of the sim. Used for boundary conditions
    grid_info: GridInfo
    bcm: BoundaryConditionManager
    n_variable_dimensions: np.int64 # Number of spatial dimensions in consideration
    simulation_params: SimParams
    metric: Metric
    current_time: np.float64 = 0.0

    def __init__(self, primitive_tensor: npt.NDArray, a_grid_info: GridInfo, a_bcm: BoundaryConditionManager, sim_params: SimParams, a_metric: Metric, starting_time: np.float64 = 0):
        n_variable_dimensions = a_grid_info.NCells.shape[0]
        assert(n_variable_dimensions == len(a_bcm.left_bcs))
        self.n_variable_dimensions = n_variable_dimensions 
        assert(primitive_tensor.ndim == (n_variable_dimensions+1)) # +1 from the variable index
        n_variables = primitive_tensor.shape[-1] # Assuming that variables are the last index 
        assert(n_variables >=3) # Need density, at least 1 velocity, and pressure 
        assert(n_variables-2==n_variable_dimensions) # Number of velocities should equal the spatial dimension
        self.simulation_params = sim_params
        self.grid_info = a_grid_info 
        self.bcm = a_bcm
        self.metric = a_metric
        U_unweighted_initial = self.primitive_to_conservative(primitive_tensor) 
        padding = self.simulation_params.spatial_integration.pad_width()
        # Make sure that the last index gets skipped, since that's associated with the variables
        pad_width = [(padding,padding)]*(U_unweighted_initial.ndim-1) + [(0,0)]
        self.U_initial_unweighted_padded = np.pad(U_unweighted_initial, pad_width, initial_value_boundary_padding) 
        self.U = self.metric.weight_system(U_unweighted_initial,  self.grid_info, WeightType.CENTER)
        self.current_time = starting_time

    def index_conservative_var(self, U_cart: npt.ArrayLike, var_type: ConservativeIndex): 
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = ConservativeIndex.X_MOMENTUM_DENSITY
            case 2:
                max_allowed_index = ConservativeIndex.Y_MOMENTUM_DENSITY 
            case _:
                raise Exception("Unimplemented simulation dimension") 
        if(var_type.value <= max_allowed_index.value):
            return U_cart[...,var_type.value]
        raise Exception("Trying to index momentum that is larger than the imension of the problem") 

    def index_primitive_var(self, primitive: npt.NDArray, var_type: PrimitiveIndex):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = PrimitiveIndex.X_VELOCITY
            case 2:
                max_allowed_index = PrimitiveIndex.Y_VELOCITY
            case _:
                raise Exception("Unimplemented simulation dimension")
        if(var_type.value <= max_allowed_index.value):
            return primitive[..., var_type.value] 
        raise Exception("Trying to index momentum that is larger than the imension of the problem")

    def primitive_to_conservative(self, W: npt.ArrayLike) -> npt.NDArray[np.float64]:
        match self.n_variable_dimensions:
            case 1:
                rho = self.index_primitive_var(W, PrimitiveIndex.DENSITY)
                x_velocity = self.index_primitive_var(W, PrimitiveIndex.X_VELOCITY)
                E = rho * self.internal_energy_primitive(W) + 0.5*rho*np.power(x_velocity,2)
                return np.stack([rho, E, rho*x_velocity], axis= self.n_variable_dimensions) 
            case 2:
                rho = self.index_primitive_var(W, PrimitiveIndex.DENSITY)
                x_velocity = self.index_primitive_var(W, PrimitiveIndex.X_VELOCITY)
                y_velocity = self.index_primitive_var(W, PrimitiveIndex.Y_VELOCITY)
                E = rho * self.internal_energy_primitive(W) + 0.5*rho*(np.power(x_velocity,2)+np.power(y_velocity,2))
                return np.stack([rho, E, rho*x_velocity, rho*y_velocity], axis=self.n_variable_dimensions)
            case _:
                raise Exception("Unimplemented simulation dimension")

    def internal_energy_primitive(self,W: npt.ArrayLike) -> npt.NDArray[np.float64]:
        match self.n_variable_dimensions:
            case 1 | 2:
                pressure = self.index_primitive_var(W, PrimitiveIndex.PRESSURE)
                density = self.index_primitive_var(W, PrimitiveIndex.DENSITY)
                return pressure /( (self.simulation_params.gamma-1) * density)  
            case _:
                raise Exception("Unimplemented simulation dimension")
            
    def internal_enthalpy_primitive(self,W: npt.ArrayLike) -> npt.NDArray[np.float64]:
        match self.n_variable_dimensions:
            case 1 | 2:
                pressure = self.index_primitive_var(W, PrimitiveIndex.PRESSURE)
                density = self.index_primitive_var(W, PrimitiveIndex.DENSITY)
                internal_energy  =self.internal_energy_primitive(W)
                return 1 + internal_energy + pressure/density
            case _:
                raise Exception("Unimplemented simulation dimension")            

    def conservative_to_primitive(self, U_cart: npt.ArrayLike) -> npt.NDArray[np.float64]:
        # TODO: Replace this with root finding method to acomodate SR
        match self.n_variable_dimensions:
            case 1:
                rho = self.index_conservative_var(U_cart,ConservativeIndex.DENSITY)
                pressure = self.equation_of_state_conservative(U_cart)
                assert(np.all(rho!=0))
                return np.stack([rho,  pressure, self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY)/ rho], axis=1) 
            case 2:
                rho = self.index_conservative_var(U_cart,ConservativeIndex.DENSITY)
                pressure = self.equation_of_state_conservative(U_cart)
                assert(np.all(rho!=0))
                return np.stack([rho,  pressure, 
                    self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY)/ rho,
                    self.index_conservative_var(U_cart, ConservativeIndex.Y_MOMENTUM_DENSITY)/ rho], axis=2) 
            case _:
                raise Exception("Unimplemented simulation dimension")

    def equation_of_state_conservative(self, U_cart: npt.ArrayLike) -> npt.NDArray[np.float64]:
        match self.n_variable_dimensions:
            case 1 | 2:
                e = np.clip(self.internal_energy_conservative(U_cart), a_min=1E-9, a_max=None)
                assert np.all(e>=0)
                return (self.simulation_params.gamma-1)*self.index_conservative_var(U_cart,ConservativeIndex.DENSITY)*e
            case _:
                raise Exception("Unimplemented simulation dimension")

    def internal_energy_conservative(self, U_cart: npt.ArrayLike) -> npt.NDArray[np.float64]:
        match self.n_variable_dimensions:
            case 1:
                rho = self.index_conservative_var(U_cart,ConservativeIndex.DENSITY)
                assert(np.all(rho!=0))
                v = self.index_conservative_var( U_cart, ConservativeIndex.X_MOMENTUM_DENSITY)/rho
                E = self.index_conservative_var(U_cart, ConservativeIndex.ENERGY)
                # Total energy
                #   E = \rho *e + 0.5*\rho * v**2
                # Inverting 
                #  E/rho-0.5 v**2 = e
                e = E/rho-0.5*np.power(v,2)
                return e
            case 2:
                rho = self.index_conservative_var(U_cart,ConservativeIndex.DENSITY)
                assert(np.all(rho!=0))
                v_x = self.index_conservative_var( U_cart, ConservativeIndex.X_MOMENTUM_DENSITY)/rho
                v_y = self.index_conservative_var( U_cart, ConservativeIndex.Y_MOMENTUM_DENSITY)/rho
                E = self.index_conservative_var(U_cart, ConservativeIndex.ENERGY)
                # Total energy
                #   E = \rho *e + 0.5*\rho * (v_x**2+v_y**2)
                # Inverting to isolate e 
                e = E/rho-0.5*(np.power(v_x,2) + np.power(v_y,2))
                return e 
            case _:
                raise Exception("Unimplemented simulation dimension")

    def flux_from_conservative(self, U_cart: npt.ArrayLike, primitive: npt.ArrayLike, spatial_index: int=0) -> npt.NDArray[np.float64]:
        match self.n_variable_dimensions:
            case 1:
                # F = (\rho v, \rho v^2 +P, (E+P) v)
                F_0 = self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY)
                F_1 = (self.index_conservative_var(U_cart, ConservativeIndex.ENERGY)+self.index_primitive_var(primitive, PrimitiveIndex.PRESSURE))*self.index_primitive_var(primitive,PrimitiveIndex.X_VELOCITY)
                F_2 = F_0*self.index_primitive_var(primitive, PrimitiveIndex.X_VELOCITY)+self.index_primitive_var(primitive, PrimitiveIndex.PRESSURE)
                return np.stack([F_0,F_1,F_2], axis=1)
            case 2:
                match spatial_index:
                    case  0: # x flux
                        # F = (\rho v_x, (E+P) v_x, \rho v_x^2 +P, \rho v_x v_y)
                        F_0 = self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY)
                        F_1 = (self.index_conservative_var(U_cart, ConservativeIndex.ENERGY)+self.index_primitive_var(primitive, PrimitiveIndex.PRESSURE))*self.index_primitive_var(primitive,PrimitiveIndex.X_VELOCITY)
                        F_2 = F_0*self.index_primitive_var(primitive, PrimitiveIndex.X_VELOCITY)+self.index_primitive_var(primitive, PrimitiveIndex.PRESSURE)
                        F_3 = F_0*self.index_primitive_var(primitive, PrimitiveIndex.Y_VELOCITY)
                        return np.stack([F_0,F_1,F_2,F_3], axis=2)
                    case 1: # y flux
                        # G = (\rho v_y, (E+P) v_y, \rho v_y^2 +P, \rho v_x v_y)
                        G_0 = self.index_conservative_var(U_cart, ConservativeIndex.Y_MOMENTUM_DENSITY)
                        G_1 = (self.index_conservative_var(U_cart, ConservativeIndex.ENERGY)+self.index_primitive_var(primitive, PrimitiveIndex.PRESSURE))*self.index_primitive_var(primitive,PrimitiveIndex.Y_VELOCITY)
                        G_2 = G_0*self.index_primitive_var(primitive, PrimitiveIndex.Y_VELOCITY)+self.index_primitive_var(primitive, PrimitiveIndex.PRESSURE)
                        G_3 = G_0*self.index_primitive_var(primitive, PrimitiveIndex.X_VELOCITY)
                        return np.stack([G_0,G_1,G_2,G_3], axis=2)
                    case _:
                        raise Exception("Invalid flux direction")
                
            case _:
                raise Exception("Unimplemented simulation dimension")

    def sound_speed(self, W: npt.ArrayLike):
        pressure = self.index_primitive_var(W, PrimitiveIndex.PRESSURE)
        density = self.index_primitive_var(W, PrimitiveIndex.DENSITY)
        return np.sqrt(self.simulation_params.gamma* pressure / density)

    def update(self) -> tuple[np.float64, npt.NDArray]:
        # Undo scaling for input
        U_cartesian = self.metric.unweight_system(self.U, self.grid_info, WeightType.CENTER)
        dt, state_update_1 = self.LinearUpdate(U_cartesian)
        U_1 = self.U+dt*state_update_1
        match self.simulation_params.time_integration:
            case TimeUpdateType.EULER:
                self.current_time  += dt
                self.U = U_1
                return self.current_time, self.U
            case TimeUpdateType.RK3:
                U_scaled_1 = self.metric.unweight_system(U_1, self.grid_info, WeightType.CENTER)
                _, state_update_2 = self.LinearUpdate(U_scaled_1)
                U_2 = (3/4)*self.U+(1/4)*U_1+(1/4)*dt*state_update_2 
                U_scaled_2 = self.metric.unweight_system(U_2, self.grid_info, WeightType.CENTER)
                _, state_update_3 = self.LinearUpdate(U_scaled_2)
                self.U = (1/3)*self.U+(2/3)*U_2+(2/3)*dt*state_update_3
                self.current_time += dt
                return self.current_time, self.U
            case _:
                raise Exception("Unimplemented TimeUpdateType Method")

    def M_dot(self):
        r_center = self.grid_info.construct_grid_centers(0) 
        unweighted_U = self.metric.unweight_system(self.U, self.grid_info, WeightType.CENTER)
        W = self.conservative_to_primitive(unweighted_U)
        return self.index_primitive_var(W, PrimitiveIndex.DENSITY)*self.index_primitive_var(W,PrimitiveIndex.X_VELOCITY)*np.power(r_center,2)
 
    def pad_unweighted_array(self, var:npt.ArrayLike):
        # Augment the array to incorporate the BCs
        # Input array is Cartesian (unweighted)
        # Returns the padded, Cartesian array 
        # Based on https://github.com/numpy/numpy/blob/main/numpy/lib/_arraypad_impl.py#L546-L926
        # Can't just use default np.pad since function signature of custom function for np.pad is not conducive for the FIXED boundary condition

        # Define padding for each axis
        padding = self.simulation_params.spatial_integration.pad_width()
        # Make sure that the last index gets skipped, since that's associated with the variables
        og_pad_width = [(padding,padding)]*(var.ndim-1) + [(0,0)]

        pad_width = np.asarray(og_pad_width)

        if not pad_width.dtype.kind == 'i':
            raise TypeError('`pad_width` must be of integral type.')

        # Broadcast to shape (array.ndim, 2)
        # Probably shouldn't be using internal numpy function calls, but it is what it is...
        pad_width = _as_pairs(pad_width, var.ndim, as_index=True)

        padded = np.pad(var, pad_width)
        assert(self.U_initial_unweighted_padded.shape == padded.shape)
        # And apply along each axis

        for axis in range(0,padded.ndim):
            # Iterate using ndindex as in apply_along_axis, but assuming that
            # function operates inplace on the padded array.

            # view with the iteration axis at the end
            view = np.moveaxis(padded, axis, -1)
            initial_view = np.moveaxis(self.U_initial_unweighted_padded,axis, -1)

            # compute indices for the iteration axes, and append a trailing
            # ellipsis to prevent 0d arrays decaying to scalars (gh-8642)
            inds = ndindex(view.shape[:-1])
            inds = (ind + (Ellipsis,) for ind in inds)
            if(axis !=  len(og_pad_width)-1):
                # Grab the B/Cs for this particular dimension
                left_bc, right_bc = self.bcm.get_boundary_conds(axis)
                for ind in inds:
                    boundary_padding(view[ind], pad_width[axis], axis, initial_view[ind], left_bc, right_bc)
        return padded

    def LinearUpdate(self, U_cart: npt.ArrayLike): 
        match self.simulation_params.spatial_integration.method:
            case SpatialUpdateType.FLAT:
                U_padded_cart = self.pad_unweighted_array(U_cart)
                W_padded_cart = self.conservative_to_primitive(U_padded_cart)
                possible_dt = []
                state_update = np.zeros(U_cart.shape)
                for dim in range(self.n_variable_dimensions):
                    flux_change, alpha_plus, alpha_minus = self.spatial_derivative(U_padded_cart,W_padded_cart, dim)
                    possible_dt.append(self.calc_dt(alpha_plus, alpha_minus))
                    state_update += flux_change
                dt = max(possible_dt)
                if(self.simulation_params.include_source):
                    # TODO: INdex variables correctly to multi dimensional prior to passing to SOurceTerm
                    slices = [slice(1,-1, None)]*W_padded_cart.ndim # Remove ghost cells from padded grid
                    slices += [slice(None)] # Select all of the variables 
                    state_update += self.SourceTerm(W_padded_cart[slices])
                return dt, state_update
            case _:
                raise Exception("Unimplemented Spatial Update")

    def spatial_derivative(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, spatial_index: np.uint = 0) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # Assuming that U is Cartesian. Cell_scaling is for the fluxes
        cell_flux = self.flux_from_conservative(U_padded_cart, W_padded_cart, spatial_index)
        alpha_minus, alpha_plus = self.alpha_plus_minus(W_padded_cart)
        alpha_sum = alpha_minus+alpha_plus 
        assert(np.all(alpha_sum != 0))
        alpha_prod = alpha_minus*alpha_plus
        # Bunch of .T because numpy broadcasting rules
        match U_padded_cart.ndim-1:
            case 1:
                # Now the plus branch
                left_cell_flux_plus = cell_flux[1:-1,:].T 
                left_conserve_plus = U_padded_cart[1:-1,:].T
                right_cell_flux_plus =cell_flux[2:,:].T  
                right_conserve_plus = U_padded_cart[2:,:].T
                # Now the minus branch
                left_cell_flux_minus = cell_flux[:-2,:].T 
                left_conserve_minus = U_padded_cart[:-2,:].T
                right_cell_flux_minus =cell_flux[1:-1,:].T  
                right_conserve_minus = U_padded_cart[1:-1,:].T
            case 2:
                # Now the plus branch
                left_cell_flux_plus = cell_flux[1:-1,1:-1,:].T 
                left_conserve_plus = U_padded_cart[1:-1,1:-1,:].T
                right_cell_flux_plus =cell_flux[2:,2:,:].T  
                right_conserve_plus = U_padded_cart[2:,2:,:].T
                # Now the minus branch
                left_cell_flux_minus = cell_flux[:-2,:-2,:].T 
                left_conserve_minus = U_padded_cart[:-2,:-2,:].T
                right_cell_flux_minus =cell_flux[1:-1,1:-1,:].T  
                right_conserve_minus = U_padded_cart[1:-1,1:-1,:].T
            case _:
                raise Exception("Unimplemented spatial dimension")
        cell_flux_plus_half = (alpha_plus*left_cell_flux_plus+ alpha_minus*right_cell_flux_plus
                               -alpha_prod*(right_conserve_plus -left_conserve_plus))/alpha_sum 
        cell_flux_minus_half = (alpha_plus*left_cell_flux_minus+ alpha_minus*right_cell_flux_minus
                                -alpha_prod*(right_conserve_minus-left_conserve_minus))/alpha_sum 
        weights = self.metric.cell_weights(self.grid_info, WeightType.EDGE) 
        slices_plus_half = [slice(1,None, None)]*weights.ndim
        slices_plus_half = tuple(slices_plus_half)
        slices_minus_half = [slice(None,-1, None)]*weights.ndim
        slices_minus_half = tuple(slices_minus_half)
        cell_flux_plus_half_rescaled = cell_flux_plus_half* weights[slices_plus_half]
        cell_flux_minus_half_rescaled = cell_flux_minus_half* weights[slices_minus_half]
        # cell_flux_plus_half_rescaled = cell_flux_plus_half* weights[1:,...]
        # cell_flux_minus_half_rescaled = cell_flux_minus_half* weights[:-1,...]         
        return -(cell_flux_plus_half_rescaled.T-cell_flux_minus_half_rescaled.T)/self.grid_info.delta()[spatial_index], alpha_plus, alpha_minus

    def alpha_plus_minus(self, primitives: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        c_s = self.sound_speed(primitives)
        lambda_plus = primitives[...,PrimitiveIndex.X_VELOCITY.value]+c_s
        lambda_minus = primitives[...,PrimitiveIndex.X_VELOCITY.value]-c_s
        # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for + and MAX(0,-left, -right) for - 
        zeros_shape = [dim-2 for dim in lambda_plus.shape]
        zeros = np.zeros(zeros_shape)
        # Grab the left and right speeds from the padded array
        match zeros.ndim:
            case 1:
                lambda_plus_left = lambda_plus[:-2]
                lambda_plus_right = lambda_plus[2:]
                lambda_minus_left = lambda_minus[:-2]
                lambda_minus_right = lambda_minus[2:]
            case 2:
                lambda_plus_left = lambda_plus[:-2,:-2]
                lambda_plus_right = lambda_plus[2:, 2:]
                lambda_minus_left = lambda_minus[:-2, :-2]
                lambda_minus_right = lambda_minus[2:, 2:]
            case 3:
                raise Exception("Unimplemented dimension")
        # First, the plus case   
        alpha_plus_candidates = np.stack([zeros, lambda_plus_left, lambda_plus_right], axis=zeros.ndim)    
        # Calculate max across each row
        alpha_plus = np.max(alpha_plus_candidates, axis=zeros.ndim)
        alpha_minus_candidates = np.stack([zeros, -lambda_minus_left, -lambda_minus_right], axis=zeros.ndim)    
        alpha_minus = np.max(alpha_minus_candidates, axis=zeros.ndim)
        return (alpha_minus, alpha_plus)
    
    def StressEnergyTensor(self,W_cart_unpadded:npt.ArrayLike):
        metric = self.get_metric_product(self.grid_info, WhichCacheTensor.METRIC, WeightType.CENTER, use_cache=True).array
        rho  = self.index_primitive_var(W_cart_unpadded, PrimitiveIndex.DENSITY.value)
        pressure  = self.index_primitive_var(W_cart_unpadded, PrimitiveIndex.PRESSURE.value)
        four_vel_shape  = [*W_cart_unpadded.shape]
        four_vel_shape[-1]  = four_vel_shape[-1]+1 # Add one for 0 component
        four_velocities  = np.zeros (four_vel_shape)
        four_velocities[...,1:]  =W_cart_unpadded[...,2:] # spatial components
        four_velocities[...,0] = 1 # 0th component
        u_u  = np.zeros(metric.shape) # Shape of (grid_size, first, secon)
        np.tensordot(four_velocities,four_velocities, axis=0)
        # Help from Gemini . Prompt:  I have a numpy array of shape (10, 10, 2). I want to take the outer product along the last axis and end up with an array of shape (10,10,2,2) . Asked for generalization for einsum
        # What it does: Takes the outer product on the last index, then moves the two indices to the front (to be compatible with the ordering of the metric field)
        u_u = np.einsum('...k,...l->kl...', a, b)
        return rho*internal_enthalpy_primitive(W)*u_u +pressure*metric

    def SourceTerm(self,W:npt.ArrayLike):
        #TODO: Modify this so that Bondi problem still works
        # Assumes that W is the array of Cartesian primitive variables. Needs to be unpadded due to construct_grid_centers call
        grid_centers = self.grid_info.mesh_grid(WeightType.CENTER)
        # primitive variables 
        # W = (\rho, v, P)
        match self.n_variable_dimensions:
            case 1:
                S_0 = np.zeros(grid_centers.shape)
                pressure = self.index_primitive_var(W, PrimitiveIndex.PRESSURE.value)
                density =  self.index_primitive_var(W, PrimitiveIndex.DENSITY.value)
                x_vel = self.index_primitive_var(W, PrimitiveIndex.X_VELOCITY.value)
                S_1 = 1.0*(2*pressure*grid_centers-density*self.simulation_params.GM)
                S_2 = 1.0*(-density*x_vel*self.simulation_params.GM) 
                return np.stack([S_0,S_1,S_2], axis=1) 
            case _:
                raise Exception("Umimplemented spatial dimension for Source Term")

    def calc_dt(self, alpha_plus: npt.ArrayLike, alpha_minus:npt.ArrayLike):
        max_alpha = np.max( [alpha_plus, alpha_minus]) 
        return self.simulation_params.Courant*np.min(self.grid_info.delta())/max_alpha

# def minmod(x: npt.ArrayLike, y: npt.ArrayLike, z: npt.ArrayLike):
#     sgn_x = np.sign(x)
#     sgn_y = np.sign(y)
#     sgn_z = np.sign(z)
#     min_candidates = np.min(np.abs(np.stack([x,y,z], axis=1)), axis=1)
#     output = (1/4)*np.abs(sgn_x+sgn_y)*(sgn_x+sgn_z)*min_candidates
#     assert(output.shape == x.shape)
#     return output
#



if __name__ == "__main__":
    pass
