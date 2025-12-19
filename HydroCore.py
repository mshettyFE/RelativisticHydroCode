import numpy.typing as nptCOre
from enum import Enum
import numpy as np
from GridInfo import GridInfo, WeightType
from UpdateSteps import TimeUpdateType,SpatialUpdate, SpatialUpdateType
from BoundaryManager import BoundaryConditionManager, BoundaryCondition
from metrics.Metric import Metric
from metrics.Metric import METRIC_VARIABLE_INDEX , WhichCacheTensor
from CommonClasses import *
from EquationOfState import *
from scipy.optimize import newton
from typing import Self

class SimulationState:
    U: npt.ArrayLike # Conservative Variables 
    U_initial_unweighted_padded: npt.ArrayLike # Initial conditions of the sim. Used for boundary conditions
    primitive_previous: npt.ArrayLike
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
        self.last_primitive_velocity_index = self.ending_primitive_velocity_index()
        self.last_conservative_velocity_index = self.ending_conservative_velocity_index()
        self.simulation_params = sim_params
        self.primitive_previous = primitive_tensor
        self.grid_info = a_grid_info 
        self.bcm = a_bcm
        self.metric = a_metric
        U_unweighted_initial = self.primitive_to_conservative(primitive_tensor) 
         # No padding along variable index
        v_mag_2_fixed = self.metric.three_vector_mag_squared(self.primitive_previous[...,PrimitiveIndex.X_VELOCITY.value:], self.grid_info, WeightType.CENTER, self.simulation_params)
        shifted_velocity_fixed = self.metric.shift_vector(self.grid_info, self.simulation_params)
        alpha_fixed = self.metric.get_metric_product(self.grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, self.simulation_params).array
        inverse_metric_fixed = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.INVERSE_METRIC, WeightType.CENTER, self.simulation_params).array
        beta_fixed = self.metric.get_metric_product(self.grid_info , WhichCacheTensor.BETA,  WeightType.CENTER, self.simulation_params).array

        # Copy the edge values for initial padding, ignoring variable index. Used for FIXED B/Cs
        self.U_initial_unweighted_padded =  self.generate_fixed_padding(U_unweighted_initial, VariableSet.CONSERVATIVE)
        self.W_initial_unweighted_padded = self.generate_fixed_padding(primitive_tensor, VariableSet.PRIMITIVE)
        self.v_mag2_fixed_padded = self.generate_fixed_padding(v_mag_2_fixed, VariableSet.SCALAR)
        self.shifted_velocity_fixed_padding = self.generate_fixed_padding(shifted_velocity_fixed, VariableSet.VECTOR)
        self.alpha_fixed_padded = self.generate_fixed_padding(alpha_fixed, VariableSet.SCALAR)
        self.inverse_metric_fixed_padded = self.generate_fixed_padding(inverse_metric_fixed, VariableSet.METRIC)
        self.inverse_beta_fixed_padded = self.generate_fixed_padding(beta_fixed, VariableSet.VECTOR) 

        assert(self.W_initial_unweighted_padded.shape == self.U_initial_unweighted_padded.shape)
        self.U = self.metric.weight_system(U_unweighted_initial,  self.grid_info, WeightType.CENTER, self.simulation_params)
        self.current_time = starting_time
        # Purely spatial dirac delta field for flux tensor calculation
        spatial_dims = [self.n_variable_dimensions]*2+[*self.U_initial_unweighted_padded[...,0].shape]
        self.dirac_delta_constant = np.zeros(spatial_dims) ## (grisize, spatial, spatial)
        for i in range(self.n_variable_dimensions):
            self.dirac_delta_constant [i,i,...] = 1

    def ending_conservative_velocity_index(self):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = ConservativeIndex.X_MOMENTUM_DENSITY
            case 2:
                max_allowed_index = ConservativeIndex.Y_MOMENTUM_DENSITY 
            case 3:
                max_allowed_index = ConservativeIndex.Z_MOMENTUM_DENSITY 
        return max_allowed_index

    def ending_primitive_velocity_index(self):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = PrimitiveIndex.X_VELOCITY
            case 2:
                max_allowed_index = PrimitiveIndex.Y_VELOCITY
            case 3:
                max_allowed_index = PrimitiveIndex.Z_VELOCITY
        return max_allowed_index


    def index_conservative_var(self, U_cart: npt.ArrayLike, start_var_type: ConservativeIndex, end_var_type: ConservativeIndex=None):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = ConservativeIndex.X_MOMENTUM_DENSITY
            case 2:
                max_allowed_index = ConservativeIndex.Y_MOMENTUM_DENSITY 
            case 3:
                max_allowed_index = ConservativeIndex.Z_MOMENTUM_DENSITY 
            case _:
                raise Exception("Unimplemented simulation dimension")
        assert(start_var_type.value <= max_allowed_index.value)
        if(end_var_type is not None):
            assert(end_var_type.value <= max_allowed_index.value)
            return U_cart[...,start_var_type.value:end_var_type.value+1]
        return U_cart[...,start_var_type.value]

         
    def primitive_to_conservative(self, W: npt.ArrayLike, weight_type: WeightType = WeightType.CENTER ) -> npt.NDArray[np.float64]:
        output = np.zeros(W.shape)
        rho = index_primitive_var(W, PrimitiveIndex.DENSITY,self.n_variable_dimensions)
        pressure = index_primitive_var(W, PrimitiveIndex.PRESSURE, self.n_variable_dimensions)
        match self.simulation_params.regime:
            case WhichRegime.NEWTONIAN:
                output[...,ConservativeIndex.DENSITY.value] = rho
                velocities_squared = np.power(W[...,PrimitiveIndex.X_VELOCITY.value:],2)
                v_mag_2 = np.sum(velocities_squared, axis=tuple(range(PrimitiveIndex.X_VELOCITY.value, velocities_squared.ndim))).T
                tau = rho* equation_of_state_primitive(self.simulation_params, pressure, rho)+0.5*rho*v_mag_2
                output[...,ConservativeIndex.TAU.value] =  tau
                output[..., ConservativeIndex.X_MOMENTUM_DENSITY.value:] = (rho.T*W[...,PrimitiveIndex.X_VELOCITY.value:].T).T
            case WhichRegime.RELATIVITY:
                enthalpy = internal_enthalpy_primitive(W, self.simulation_params, self.n_variable_dimensions)
                velocities = W[...,PrimitiveIndex.X_VELOCITY.value:]
                alpha  = self.metric.get_metric_product(self.grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, self.simulation_params) 
                boost = self.metric.W(alpha, velocities, self.grid_info,weight_type, self.simulation_params)
                D = rho*boost
                output[...,ConservativeIndex.DENSITY.value] = D
                output[...,ConservativeIndex.TAU.value] = rho*enthalpy*np.power(boost,2)-pressure-D 
                first = rho*enthalpy*np.power(boost,2) 
                secon = (W[...,PrimitiveIndex.X_VELOCITY.value:].T)
                output[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:] =(first.T *secon).T
            case _:
                raise Exception("Unimplemented relativistic regime")
        return output
    
    def conservative_to_primitive(self, U_cart_padded: npt.ArrayLike) -> npt.NDArray[np.float64]:
        # Root finder will use NR to calculate pressure
        # Once converged, use equations in Marti et. al to spit out estimated density and velocities
        # Bundle up in primitive variable tensor
        # Returns primitive array without any of the ghost cells  since that would require rewriting a bunch of the Metric class stuff that is too much of a pain
        # NOTE: Make sure that you pad the output appropriately
         # Remove ghost cells from padded grid
        slices = [slice(1,-1, None)]*(U_cart_padded.ndim-1)
        slices += [slice(None)] # Select all of the variables 
        slices = tuple(slices)
        U_cart_padded  = U_cart_padded[slices]
        match self.simulation_params.regime:
            case WhichRegime.NEWTONIAN:
                output = np.zeros(U_cart_padded.shape)
                rho = self.index_conservative_var(U_cart_padded,ConservativeIndex.DENSITY)
                assert(np.all(rho!=0))
                flux = self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY, self.last_conservative_velocity_index)
                velocities = (flux.T/rho.T).T
                velocities_squared = np.power(velocities,2)
                v_mag_2 = np.sum(velocities_squared, axis=-1).T
                E = self.index_conservative_var(U_cart_padded, ConservativeIndex.TAU)
                unclipped_e =  E/rho-0.5*(v_mag_2)
                e = np.clip(unclipped_e, a_min=1E-9, a_max=None)
                assert np.all(e>=0)
                pressure  = equation_of_state_epsilon(self.simulation_params, e,rho )
                flux = self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY, self.last_conservative_velocity_index)
                velocities = (flux.T/rho.T).T
                output = np.zeros(U_cart_padded.shape)
                output[...,PrimitiveIndex.DENSITY.value] = rho
                output[..., PrimitiveIndex.X_VELOCITY.value:] = velocities
                output[...,PrimitiveIndex.PRESSURE.value] =  pressure
                return output
            case WhichRegime.RELATIVITY:
                args = (U_cart_padded, self)
                initial_guess = index_primitive_var(self.primitive_previous, PrimitiveIndex.PRESSURE, self.n_variable_dimensions)
                initial_guess = np.maximum(initial_guess, 1e-10)
                recovered_guess, converge, zero_der = newton(self.root_finding_func, initial_guess,args = args, full_output=True)
                assert(np.all(converge))
                out = self.construct_primitives_from_guess(recovered_guess, U_cart_padded)
                return out
            case _:
                raise Exception("Unimplemented relativistic regime")
            
    def root_finding_func(self, guess: npt.ArrayLike,
                                U_cart: npt.ArrayLike, simulation_instance: Self) -> npt.NDArray:
        flux = self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY, self.last_conservative_velocity_index)
        flux_squared = simulation_instance.metric.three_vector_mag_squared(flux, self.grid_info, WeightType.CENTER, simulation_instance.simulation_params)
        D = self.index_conservative_var(U_cart, ConservativeIndex.DENSITY)
        Tau = self.index_conservative_var(U_cart, ConservativeIndex.TAU)
        z = Tau+guess+D
        z = np.maximum(z, 1e-12)
        v_mag_2 = flux_squared/np.power(z,2)
        v_mag_2 = np.minimum(v_mag_2, 1.0 - 1e-10)
        W2 = np.power(1-v_mag_2,-1)
        W = np.sqrt(W2)
        epsilon = (Tau+D*(1-W)+guess*(1.0-W2))/(D*W)
        epsilon = np.maximum(epsilon, 1e-10)
        rho = D/W
        guess_pressure = equation_of_state_epsilon(self.simulation_params, epsilon,rho )
        out  = guess_pressure - guess
        return out

    def construct_primitives_from_guess(self, guess:npt.ArrayLike,
                                        U_cart: npt.ArrayLike) -> npt.NDArray:
        guess = np.maximum(guess, 1e-12) # Clamp guess to be physically valid
        output = np.zeros(U_cart.shape)
        flux = self.index_conservative_var(U_cart, ConservativeIndex.X_MOMENTUM_DENSITY, self.last_conservative_velocity_index)
        flux_squared = self.metric.three_vector_mag_squared(flux, self.grid_info, WeightType.CENTER, self.simulation_params)
        D = self.index_conservative_var(U_cart, ConservativeIndex.DENSITY)
        Tau = self.index_conservative_var(U_cart, ConservativeIndex.TAU)
        z = Tau+guess+D
        z = np.maximum(z, 1e-12)
        v_mag_2 = flux_squared/np.power(z,2)
        v_mag_2 = np.minimum(v_mag_2, 1.0 - 1e-10)
        W2 = np.power(1-v_mag_2,-1)
        W = np.sqrt(W2)
        rho = D/W
        velocities = ((flux.T)/(z.T)).T
        rho = np.maximum(rho, 1e-12)
        velocities = np.clip(velocities, -0.99999, 0.99999)
        output[...,PrimitiveIndex.DENSITY.value] = rho
        output[...,PrimitiveIndex.X_VELOCITY.value:] = velocities
        output[...,PrimitiveIndex.PRESSURE.value] = guess
        return output
        
    def update(self, which_axes : tuple = ()) -> tuple[np.float64, npt.NDArray]:
        # Undo scaling for input
        U_cartesian = self.metric.unweight_system(self.U, self.grid_info, WeightType.CENTER, self.simulation_params)
        dt, state_update_1, primitives = self.LinearUpdate(U_cartesian, which_axes)
        U_1 = self.U+dt*state_update_1
#        assert((index_primitive_var(primitives, PrimitiveIndex.PRESSURE, self.n_variable_dimensions)>=0).all())
        self.primitive_previous = primitives
        match self.simulation_params.time_integration:
            case TimeUpdateType.EULER:
                self.current_time  += dt
                self.U = U_1
            case TimeUpdateType.RK3:
                U_scaled_1 = self.metric.unweight_system(U_1, self.grid_info, WeightType.CENTER, self.simulation_params)
                _, state_update_2, primitives = self.LinearUpdate(U_scaled_1, which_axes)
                self.primitive_previous = primitives
                U_2 = (3/4)*self.U+(1/4)*U_1+(1/4)*dt*state_update_2 
                U_scaled_2 = self.metric.unweight_system(U_2, self.grid_info, WeightType.CENTER, self.simulation_params)
                _, state_update_3, primitives = self.LinearUpdate(U_scaled_2, which_axes)
                self.primitive_previous = primitives
                self.U = (1/3)*self.U+(2/3)*U_2+(2/3)*dt*state_update_3
                self.current_time += dt
            case _:
                raise Exception("Unimplemented TimeUpdateType Method")
        return self.current_time, self.U


    def M_dot(self):
        r_center = self.grid_info.construct_grid_centers(0) 
        unweighted_U = self.metric.unweight_system(self.U, self.grid_info, WeightType.CENTER, self.simulation_params)
        W = self.conservative_to_primitive(unweighted_U)
        return index_primitive_var(W, PrimitiveIndex.DENSITY,self.n_variable_dimensions)*index_primitive_var(W,PrimitiveIndex.X_VELOCITY,self.n_variable_dimensions)*np.power(r_center,2)
 

    def axis_to_vector_index_map(self, var_type: VariableSet) -> dict:
        match var_type:
            case VariableSet.CONSERVATIVE:
                axis_to_vel_map = {
                    0: ConservativeIndex.X_MOMENTUM_DENSITY.value,
                    1: ConservativeIndex.Y_MOMENTUM_DENSITY.value,
                    2: ConservativeIndex.Z_MOMENTUM_DENSITY.value
                }
            case VariableSet.PRIMITIVE:
                axis_to_vel_map = {
                    0: PrimitiveIndex.X_VELOCITY.value,
                    1: PrimitiveIndex.Y_VELOCITY.value,
                    2: PrimitiveIndex.Z_VELOCITY.value
                }
            case VariableSet.VECTOR:
                axis_to_vel_map = {
                    0: 0,
                    1: 1,
                    2: 2
                }
            case VariableSet.SCALAR:
                axis_to_vel_map = {}
            case VariableSet.METRIC:
                axis_to_vel_map = {}
            case _:
                raise Exception("Unimplemented variable type for axis to vector index mapping")
        return axis_to_vel_map
    
    def excluded_padding_axes(self, array: npt.ArrayLike, var_type: VariableSet) -> list:
        match var_type:
            case VariableSet.CONSERVATIVE | VariableSet.PRIMITIVE:
                return [array.ndim - 1]
            case VariableSet.VECTOR:
                return [0] # NOTE: Hack for now since the only vectors have the variable index at the start
            case VariableSet.SCALAR:
                return []
            case VariableSet.METRIC:
                return [0,1 ]
            case _:
                raise Exception("Unimplemented variable type for excluded padding axes")


    def generate_fixed_padding(self, var, var_set: VariableSet):
        padding = self.simulation_params.spatial_integration.pad_width() # NOTE: This is just 1 for now. Would need to modify for MiniMod
        pad_width = [(padding,padding)]*(var.ndim)
        assert(self.n_variable_dimensions <= var.ndim)
        has_variables = self.n_variable_dimensions != var.ndim 
        if(has_variables):
            excluded_axes = self.excluded_padding_axes(var, var_set)
            for axis in excluded_axes:
                pad_width[axis] = (0,0)
        return np.pad(var, pad_width, "edge")

    def pad_array(self, var:npt.ArrayLike, fixed: npt.ArrayLike, var_set: VariableSet) -> npt.NDArray:
        # Can't just use default np.pad since function signature of custom function for np.pad is not conducive for the FIXED boundary condition
        assert(self.n_variable_dimensions <= var.ndim)
        has_variables = self.n_variable_dimensions != var.ndim
        padding = self.simulation_params.spatial_integration.pad_width() # NOTE: This is just 1 for now. Would need to modify for MiniMod
        pad_width = [(padding,padding)]*(var.ndim)
        included_axes = list(range(var.ndim)) # Initially include all axes
        if(has_variables):
            excluded_axes = self.excluded_padding_axes(var, var_set)
            for axis in excluded_axes:
                included_axes.remove(axis) # Skip this axis while iterating
                pad_width[axis] = (0,0)
        grab_everything = [slice(None,None,None)]*var.ndim
        # First pad the array  with zeros
        padded = np.pad(var, pad_width)
        # Select the correct fixed boundaries
        assert(fixed.shape == padded.shape)
        # Now apply boundary padding
        for axis in included_axes:
            # Grab the B/Cs for this particular dimension
            left_bc, right_bc = self.bcm.get_boundary_conds(axis)
            # Determine the pad widths for this axis
            iaxis_pad_width = pad_width[axis]
            # First select the cells that need updating
            lower_slice = grab_everything.copy() # A priori,select everything
            lower_slice[axis] = slice(0, iaxis_pad_width[0], None) # On the current axis, select only the padded portion
            lower_slice = tuple(lower_slice)
            upper_slice = grab_everything.copy()
            upper_slice[axis] = slice(-iaxis_pad_width[1], None, None) 
            upper_slice = tuple(upper_slice)
            # The values needed for zero_grad BCs at the lower boundary (ie. The first non-padded element along this axis)
            lower_zero_grad_slices = grab_everything.copy()
            lower_zero_grad_slices[axis] = slice(iaxis_pad_width[0], iaxis_pad_width[0]+1, None)
            lower_zero_grad_slices = tuple(lower_zero_grad_slices)
            upper_zero_grad_slices = grab_everything.copy()
            upper_zero_grad_slices[axis] = slice(-iaxis_pad_width[1]-1, -iaxis_pad_width[1], None)
            upper_zero_grad_slices = tuple(upper_zero_grad_slices)
            # For fixed, use the same slices, but index the fixed array
            axis_to_vel_map  = self.axis_to_vector_index_map(var_set)
            has_vector_components = len(axis_to_vel_map)>0
            if(has_vector_components): # (Only need to do this if we have vector variables)
                target_idx = axis_to_vel_map[axis]
                
                #For Reflective, scalars acts like zero grad, vectors need to flip sign
                lower_vector_reflect_slices_pad = [*lower_slice] #  Copy over the padded slices
                lower_vector_reflect_slices_pad[-1] = slice(target_idx, target_idx+1, None)# The vectors need to be singled out. Only the normal component to the boundary needs to be flipped
                lower_vector_reflect_slices_pad = tuple(lower_vector_reflect_slices_pad)

                upper_vector_reflect_slices_pad = [*upper_slice] #  Copy over the padded slices
                upper_vector_reflect_slices_pad[-1] = slice(target_idx,  target_idx+1, None)# The vectors need to be singled out. Only the normal component to the boundary needs to be flipped
                upper_vector_reflect_slices_pad = tuple(upper_vector_reflect_slices_pad)

            match left_bc:
                case BoundaryCondition.ZERO_GRAD: # Just assign the first non-padded value to the padded region
                    padded[lower_slice] = padded[lower_zero_grad_slices]
                case BoundaryCondition.FIXED: # Assign from the fixed initial array
                    padded[lower_slice] = fixed[lower_slice]
                case BoundaryCondition.REFLECTIVE:
                    # Initially assign like zero grad ( takes care of scalars)
                    padded[lower_slice] = padded[lower_zero_grad_slices]
                    if(has_vector_components):
                        # For vectors, need to flip sign
                        padded[lower_vector_reflect_slices_pad] *= -1
                case _:
                    raise Exception("Unimplemented BC")
            match right_bc:
                case BoundaryCondition.ZERO_GRAD:
                    padded[upper_slice] = padded[upper_zero_grad_slices]
                case BoundaryCondition.FIXED:       
                    padded[upper_slice] = fixed[upper_slice]
                case BoundaryCondition.REFLECTIVE:
                    # Initially assign like zero grad ( takes care of scalars)
                    padded[upper_slice] = padded[upper_zero_grad_slices]
                    if(has_vector_components):
                        # For vectors, need to flip sign
                        padded[upper_vector_reflect_slices_pad] *= -1
                case _:
                    raise Exception("Unimplemented BC")
        return padded

    def LinearUpdate(self, U_cart: npt.ArrayLike, which_axis: tuple = ()): 
        # Returns the time step needed, the change in the conservative variables, and the recovered primitives of the system prior to stepping
        match self.simulation_params.spatial_integration.method:
            case SpatialUpdateType.FLAT:
                U_padded_cart = self.pad_array(U_cart, self.U_initial_unweighted_padded, VariableSet.CONSERVATIVE)
                W_cart = self.conservative_to_primitive(U_padded_cart)
                W_padded_cart = self.pad_array(W_cart, self.W_initial_unweighted_padded, VariableSet.PRIMITIVE)
                assert(W_padded_cart.shape == U_padded_cart.shape)
                possible_dt = []
                state_update = np.zeros(U_cart.shape)
                if(len(which_axis)==0):
                    axes = tuple(range(self.n_variable_dimensions))
                else:
                    axes = which_axis
                density_flux_padded = self.density_flux(U_padded_cart, W_padded_cart, WeightType.CENTER)
                tau_flux_padded = self.tau_flux(U_padded_cart, W_padded_cart, WeightType.CENTER)
                momentum_flux_tensor_padded = self.momentum_flux_tensor(U_padded_cart,W_padded_cart, WeightType.CENTER)
                v_mag_2 = self.metric.three_vector_mag_squared(W_cart[...,PrimitiveIndex.X_VELOCITY.value:], self.grid_info, WeightType.CENTER, self.simulation_params)
                alpha = self.metric.get_metric_product(self.grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, self.simulation_params).array 
                beta = self.metric.get_metric_product(self.grid_info , WhichCacheTensor.BETA,  WeightType.CENTER, self.simulation_params).array
                inv_metric =  self.metric.get_metric_product(self.grid_info, WhichCacheTensor.INVERSE_METRIC, WeightType.CENTER, self.simulation_params).array

                v_mag_2_fixed_current_padded = self.pad_array(v_mag_2, self.v_mag2_fixed_padded, VariableSet.VECTOR)
                alpha_padded = self.pad_array( alpha, self.alpha_fixed_padded, VariableSet.SCALAR)
                beta_padded = self.pad_array( beta, self.inverse_beta_fixed_padded, VariableSet.VECTOR)
                inverse_metric_padded = self.pad_array(inv_metric , self.inverse_metric_fixed_padded, VariableSet.METRIC) #  NOTE: Going to treat metric as a scalar. Don't know how matrices should reflect...
                for dim in axes:
                    flux_change, alpha_plus, alpha_minus = self.spatial_derivative(U_padded_cart,W_padded_cart,
                                                                                   density_flux_padded, momentum_flux_tensor_padded, tau_flux_padded,
                                                                                   v_mag_2_fixed_current_padded,alpha_padded, inverse_metric_padded, beta_padded,   
                                                                                       dim)
                    possible_dt.append(self.calc_dt(alpha_plus, alpha_minus))
                    state_update += flux_change
                dt = max(possible_dt)
                slices = [slice(1,-1, None)]*(W_padded_cart.ndim-1) # Remove ghost cells from padded grid
                slices += [slice(None)] # Select all of the variables 
                slices = tuple(slices)
                primitives = W_padded_cart[slices]
                if(self.simulation_params.include_source):
                    source_term = self.SourceTerm(primitives) 
                    state_update += self.metric.weight_system(source_term, self.grid_info, WeightType.CENTER, self.simulation_params)
                return dt, state_update, primitives
            case _:
                raise Exception("Unimplemented Spatial Update")
    
    def shift_three_velocity_padded(self, three_velocities_padded: npt.ArrayLike) -> npt.NDArray:
        shift_vec = self.metric.shift_vector(self.grid_info, self.simulation_params)
        shift_vec_padded = self.pad_array(shift_vec, self.shifted_velocity_fixed_padding, VariableSet.VECTOR)
        return three_velocities_padded - shift_vec_padded.T

    def density_flux(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index)
        shifted_velocity = self.shift_three_velocity_padded(W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value:])
        D = self.index_conservative_var(U_padded_cart, ConservativeIndex.DENSITY)
        return (D.T*shifted_velocity.T).T

    def momentum_flux_tensor(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index, spatial_index) where the first indexes the coordinate and the 2nd indexes the direction
        shifted_velocity = self.shift_three_velocity_padded(W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value:])
        pressure = index_primitive_var(W_padded_cart,PrimitiveIndex.PRESSURE,self.n_variable_dimensions)
        flux = self.index_conservative_var(U_padded_cart, ConservativeIndex.X_MOMENTUM_DENSITY, self.last_conservative_velocity_index)
        pressure_tensor = pressure*self.dirac_delta_constant
        pressure_tensor = np.einsum("ij...->...ij", pressure_tensor)
        KE = np.einsum("...i,...j->...ij", flux, shifted_velocity)
        return KE+pressure_tensor
    
    def tau_flux(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index)
        Tau =  self.index_conservative_var(U_padded_cart, ConservativeIndex.TAU)
        pressure = index_primitive_var(W_padded_cart,PrimitiveIndex.PRESSURE,self.n_variable_dimensions)
        velocity = W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value:]
        shifted_velocity = self.shift_three_velocity_padded(W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value:])
        return (Tau.T*shifted_velocity.T+ pressure.T*velocity.T).T

    def spatial_derivative(self, 
                           U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, 
                           density_flux_padded: npt.ArrayLike, momentum_flux_tensor_padded: npt.ArrayLike, tau_flux_padded: npt.ArrayLike,
                           v_mag_2_padded: npt.ArrayLike, alpha_padded: npt.ArrayLike, inverse_metric_padded: npt.ArrayLike, beta_padded: npt.ArrayLike,
                             spatial_index: np.uint = 0) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # Assuming that U is Cartesian. Cell_scaling is for the fluxes
        zero  =  density_flux_padded[...,spatial_index]
        target = [*zero.shape]+[1]
        zero = zero.reshape(target)
        first  = tau_flux_padded[...,spatial_index]
        first = first.reshape(target)
        rest  = momentum_flux_tensor_padded[...,spatial_index,:]
        cell_flux = np.concatenate(
            [
                zero,
                first,
                rest
            ], 
            axis=-1)
        pressure = index_primitive_var(W_padded_cart, PrimitiveIndex.PRESSURE,self.n_variable_dimensions)
        density = index_primitive_var(W_padded_cart, PrimitiveIndex.DENSITY,self.n_variable_dimensions)
        c_s = sound_speed(self.simulation_params,pressure, density)    
        match self.simulation_params.regime:
            case WhichRegime.NEWTONIAN:
                lambda_plus = W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value+spatial_index]+c_s
                lambda_minus = W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value+spatial_index]-c_s
                alpha_minus, alpha_plus = self.alpha_plus_minus(lambda_plus, lambda_minus)
            case WhichRegime.RELATIVITY:
                # Eq. 22 of https://iopscience.iop.org/article/10.1086/303604/pdf
                beta_component = beta_padded[spatial_index,...]
                gamma_ii = inverse_metric_padded[METRIC_VARIABLE_INDEX.SPACE_1.value+spatial_index,METRIC_VARIABLE_INDEX.SPACE_1.value+spatial_index,...] 
                velocity_component = W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value+spatial_index]
                cs_2 = np.power(c_s,2)
                factor_1 = 1-v_mag_2_padded*cs_2
                factor_2 = 1-v_mag_2_padded 
                factor_3 = 1-cs_2
                prefactor = alpha_padded/factor_1
                common_factor = velocity_component*factor_3
                discriminant = factor_2*(gamma_ii*factor_1-np.power(velocity_component,2)*factor_3)
                lambda_plus = prefactor*(common_factor + c_s*np.sqrt(discriminant)) - beta_component
                lambda_minus = prefactor*(common_factor - c_s*np.sqrt(discriminant)) - beta_component
                alpha_minus, alpha_plus = self.alpha_plus_minus(lambda_plus, lambda_minus)
                assert(alpha_minus.shape == alpha_plus.shape)
        alpha_sum = alpha_minus+alpha_plus
        assert(np.all(alpha_sum != 0))
        alpha_prod = alpha_minus*alpha_plus
        slices_left = [slice(0,-1,None)]*cell_flux.ndim
        slices_left[-1] = slice(None,None,None) # Select all variables
        slices_left = tuple(slices_left)
        slices_right = [slice(1,None,None)]*cell_flux.ndim
        slices_right[-1] = slice(None,None,None) # Select all variables
        slices_right = tuple(slices_right)
        cell_flux_left = cell_flux[slices_left]
        conserve_left = U_padded_cart[slices_left]
        cell_flux_right = cell_flux[slices_right]
        conserve_right = U_padded_cart[slices_right]
        flux_interface = ((alpha_plus * cell_flux_left.T + alpha_minus * cell_flux_right.T
                      - alpha_prod * (conserve_right.T - conserve_left.T)) / (alpha_sum)).T
        weights = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.DETERMINANT, WeightType.EDGE, self.simulation_params).array
        flux_densitized = (flux_interface.T * weights.T).T
        flux_interface_left = flux_densitized[slices_left]
        flux_interface_right = flux_densitized[slices_right]
        out = (flux_interface_right - flux_interface_left)/self.grid_info.delta(spatial_index)[spatial_index]
        return out , alpha_plus, alpha_minus

    def alpha_plus_minus(self, lambda_plus: npt.ArrayLike, lambda_minus: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for + and MAX(0,-left, -right) for - 
        zeros_shape = [dim-1 for dim in lambda_plus.shape]
        zeros = np.zeros(zeros_shape)
        # Grab the left and right speeds from the padded array
        slices_left = tuple([slice(0,-1,None)]*self.n_variable_dimensions)
        slices_right = tuple([slice(1,None,None)]*self.n_variable_dimensions)
        lambda_plus_left = lambda_plus[slices_left]
        lambda_plus_right = lambda_plus[slices_right]
        lambda_minus_left = lambda_minus[slices_left]
        lambda_minus_right = lambda_minus[slices_right]
        # First, the plus case   
        alpha_plus_candidates = np.stack([zeros, lambda_plus_left, lambda_plus_right], axis=zeros.ndim)    
        # Calculate max across each row
        alpha_plus = np.max(alpha_plus_candidates, axis=zeros.ndim)
        alpha_minus_candidates = np.stack([zeros, -lambda_minus_left, -lambda_minus_right], axis=zeros.ndim)    
        alpha_minus = np.max(alpha_minus_candidates, axis=zeros.ndim)
        return (alpha_minus, alpha_plus)
    
    def StressEnergyTensor(self,W_cart_unpadded:npt.ArrayLike):
        # TODO: Change 4 vector to include normalization factor (Did for SR, need to generalize to GR)
        metric = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.METRIC, WeightType.CENTER, self.simulation_params, use_cache=True).array
        rho  = index_primitive_var(W_cart_unpadded, PrimitiveIndex.DENSITY,self.n_variable_dimensions)
        pressure  = index_primitive_var(W_cart_unpadded, PrimitiveIndex.PRESSURE,self.n_variable_dimensions)
        velocities = W_cart_unpadded[...,PrimitiveIndex.X_VELOCITY.value:] # spatial components of 4 velocity
        four_vel_shape  = [*velocities.shape]
        four_vel_shape[-1]  = four_vel_shape[-1]+1 # Add one for 0 component
        four_velocities  = np.zeros (four_vel_shape)
        four_velocities[...,0] = 1 # 0th component
        # First shift the spatial velocities to the current spacelike slice
        shifted_three_velocities = self.metric.shift_three_velocity(velocities, self.grid_info, self.simulation_params)
        four_velocities[...,1:] = shifted_three_velocities 
        # Scale by the boost
        alpha = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.ALPHA, WeightType.CENTER, self.simulation_params).array
        W = self.metric.W(alpha, velocities, self.grid_info,WeightType.CENTER, self.simulation_params)
        four_velocities[...] = (W.T*four_velocities[...].T).T
        u_u  = np.zeros(metric.shape) # Shape of (grid_size, first, secon)
        # Help from Gemini . Prompt:  I have a numpy array of shape (10, 10, 2). I want to take the outer product along the last axis and end up with an array of shape (10,10,2,2) . Asked for generalization for einsum
        # What it does: Takes the outer product on the last index, then moves the two indices to the front (to be compatible with the ordering of the metric field)
        u_u = np.einsum('...k,...l->kl...', four_velocities, four_velocities)
        inv_metric = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.INVERSE_METRIC, WeightType.CENTER, self.simulation_params, use_cache=True).array
        t_mu_nu_raised = rho*internal_enthalpy_primitive(W_cart_unpadded, self.simulation_params, self.n_variable_dimensions)*u_u +pressure*inv_metric
        return t_mu_nu_raised

    def SourceTerm(self,W:npt.ArrayLike):
        # Assumes that W is the array of Cartesian primitive variables. Needs to be unpadded due to construct_grid_centers call
        # primitive variables 
        # W = (\rho, P, v_{j})

        output = np.zeros(W.shape)
        # Hack to make Bondi work without implementing Source Term w.r.t. Metric and stress energy
        # gri  =self.grid_info.mesh_grid(WeightType.CENTER)
        # rho  =index_primitive_var(W, PrimitiveIndex.DENSITY)
        # pressure  =index_primitive_var(W, PrimitiveIndex.PRESSURE)
        # output[..., 0]  = 0 # density
        # output[..., 1]  = -rho*index_primitive_var(W,PrimitiveIndex.X_VELOCITY)*self.simulation_params.GM #energy
        # output[..., 2]  = 2*gri[0]*pressure-rho*self.simulation_params.GM # momentum
        # output[..., 3]  = 0 
        # output[..., 4]  = 0 
        # return output
        T_mu_nu_raised = self.StressEnergyTensor(W)
        metric: Metric = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.METRIC,  WeightType.CENTER, self.simulation_params).array
        partial_metric = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.PARTIAL_DER,  WeightType.CENTER, self.simulation_params).array
        partial_ln_alpha = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.PARTIAL_LN_ALPHA,  WeightType.CENTER, self.simulation_params).array
        alpha =self.metric.get_metric_product(self.grid_info, WhichCacheTensor.ALPHA,  WeightType.CENTER, self.simulation_params).array
        Christoffel_upper = self.metric.get_metric_product(self.grid_info, WhichCacheTensor.CHRISTOFFEL_UPPER0,  WeightType.CENTER, self.simulation_params).array
        rho_source = np.zeros(alpha.shape)
        tau_first = np.einsum("ij...,i...->j...", T_mu_nu_raised, partial_ln_alpha)[METRIC_VARIABLE_INDEX.TIME.value,...]
        tau_second = np.einsum("ij...,kji...->k...", T_mu_nu_raised, Christoffel_upper)[METRIC_VARIABLE_INDEX.TIME.value,...]
        assert(tau_first.shape == tau_second.shape)
        assert(alpha.shape == tau_first.shape)
        energy_source = alpha*(tau_first+tau_second)
        flux_inner_part_two = np.einsum("kij...,kl...->jil...", Christoffel_upper, metric)
        assert(partial_metric.shape == flux_inner_part_two.shape)
        source_flux_part_one = np.einsum("ij...,ijk...->...k", T_mu_nu_raised,partial_metric)
        source_flux_part_two = np.einsum("ij...,ijk...->...k", T_mu_nu_raised, flux_inner_part_two)
        source_flux = source_flux_part_one-source_flux_part_two
        output[...,ConservativeIndex.DENSITY.value] = rho_source
        output[...,ConservativeIndex.TAU.value] = energy_source
        output[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:] = source_flux[..., METRIC_VARIABLE_INDEX.SPACE_1.value:]
        return output

    def calc_dt(self, alpha_plus: npt.ArrayLike, alpha_minus:npt.ArrayLike):
        max_alpha = np.max( [alpha_plus, alpha_minus]) 
        deltas = np.asarray([np.min(self.grid_info.delta(i)) for i in range(self.n_variable_dimensions)])
        min_delta = np.min(deltas[deltas>0])
        return self.simulation_params.Courant*min_delta/max_alpha


if __name__ == "__main__":
    pass
