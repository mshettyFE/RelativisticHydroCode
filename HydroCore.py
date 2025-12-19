import numpy.typing as npt
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
        n_variables = primitive_tensor.shape[0] # Assuming that variables are the first index
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
        velocities = self.get_velocities(self.primitive_previous)
        v_mag_2_fixed = self.metric.three_vector_mag_squared(velocities, self.grid_info, WeightType.CENTER, self.simulation_params)
        shifted_velocity_fixed = self.metric.shift_vector(self.grid_info, self.simulation_params)

        alpha_fixed = self.metric.get_metric_product(self.grid_info , self.metric.construct_index(WhichCacheTensor.ALPHA, WeightType.CENTER), self.simulation_params)
        inverse_metric_fixed = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.INVERSE_METRIC, WeightType.CENTER), self.simulation_params)
        beta_fixed = self.metric.get_metric_product(self.grid_info , self.metric.construct_index(WhichCacheTensor.BETA, WeightType.CENTER) ,self.simulation_params)

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
        spatial_dims = [self.n_variable_dimensions]*2+[*self.U_initial_unweighted_padded[0,...].shape]
        self.dirac_delta_constant = np.zeros(spatial_dims) ## (grisize, spatial, spatial)
        for i in range(self.n_variable_dimensions):
            self.dirac_delta_constant [i,i,...] = 1

    def ending_conservative_velocity_index(self):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_vel_index = ConservativeIndex.X_MOMENTUM_DENSITY
            case 2:
                max_allowed_vel_index = ConservativeIndex.Y_MOMENTUM_DENSITY 
            case 3:
                max_allowed_vel_index = ConservativeIndex.Z_MOMENTUM_DENSITY 
        return max_allowed_vel_index

    def ending_primitive_velocity_index(self):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_flux_index = PrimitiveIndex.X_VELOCITY
            case 2:
                max_allowed_flux_index = PrimitiveIndex.Y_VELOCITY
            case 3:
                max_allowed_flux_index = PrimitiveIndex.Z_VELOCITY
        return max_allowed_flux_index


    def get_conservative_var(self, U_cart: npt.ArrayLike, start_var_type: ConservativeIndex):
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
        return U_cart[start_var_type.value,...]
    
    def set_conservative_var(self, U_cart: npt.ArrayLike, start_var_type: ConservativeIndex, new_values: npt.ArrayLike):
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
        if(new_values.shape[0]==1):
            assert (U_cart[start_var_type.value,...].shape[0]== new_values.shape[1])
        else:
            assert(U_cart[start_var_type.value,...].shape == new_values.shape)
        U_cart[start_var_type.value,...] = new_values
        return True

    def get_momentum_fluxes(self, U_cart: npt.ArrayLike):
        # Assumes momenta are contiguous in the conservative variable array
        return U_cart[ConservativeIndex.X_MOMENTUM_DENSITY.value:self.last_conservative_velocity_index.value+1,...]
    
    def set_momentum_fluxes(self, U_cart: npt.ArrayLike, new_values: npt.ArrayLike):
        assert(U_cart[ConservativeIndex.X_MOMENTUM_DENSITY.value:self.last_conservative_velocity_index.value+1,...].shape == new_values.shape)
        U_cart[ConservativeIndex.X_MOMENTUM_DENSITY.value:self.last_conservative_velocity_index.value+1,...] = new_values
        return True
    
    def get_specific_momentum_flux(self, U_cart: npt.ArrayLike, axis: int):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = ConservativeIndex.X_MOMENTUM_DENSITY
            case 2:
                max_allowed_index = ConservativeIndex.Y_MOMENTUM_DENSITY 
            case 3:
                max_allowed_index = ConservativeIndex.Z_MOMENTUM_DENSITY 
            case _:
                raise Exception("Unimplemented simulation dimension")
        axis_to_index_map = {
            0: ConservativeIndex.X_MOMENTUM_DENSITY.value,
            1: ConservativeIndex.Y_MOMENTUM_DENSITY.value,
            2: ConservativeIndex.Z_MOMENTUM_DENSITY.value
        }
        assert(axis in axis_to_index_map.keys())
        target_index = axis_to_index_map[axis]
        assert(target_index <= max_allowed_index.value)
        return U_cart[target_index,...]
    
    def get_primitive_var(self,  primitive: npt.NDArray, start_var_type: PrimitiveIndex):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = PrimitiveIndex.X_VELOCITY
            case 2:
                max_allowed_index = PrimitiveIndex.Y_VELOCITY
            case 3:
                max_allowed_index = PrimitiveIndex.Z_VELOCITY
            case _:
                raise Exception("Unimplemented simulation dimension")
        assert(start_var_type.value <= max_allowed_index.value)
        return primitive[start_var_type.value,...]

    def get_velocities(self, primitive: npt.ArrayLike):
        # Assumes velocities are contiguous in the primitive variable array
        return primitive[PrimitiveIndex.X_VELOCITY.value:self.last_primitive_velocity_index.value+1,...]

    def get_specific_velocity(self, primitive: npt.ArrayLike, axis: int):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = PrimitiveIndex.X_VELOCITY
            case 2:
                max_allowed_index = PrimitiveIndex.Y_VELOCITY 
            case 3:
                max_allowed_index = PrimitiveIndex.Z_VELOCITY 
            case _:
                raise Exception("Unimplemented simulation dimension")
        axis_to_index_map = {
            0: PrimitiveIndex.X_VELOCITY.value,
            1: PrimitiveIndex.Y_VELOCITY.value,
            2: PrimitiveIndex.Z_VELOCITY.value
        }
        assert(axis in axis_to_index_map.keys())
        target_index = axis_to_index_map[axis]
        assert(target_index <= max_allowed_index.value)
        return primitive[target_index,...]

    def set_primitive_var(self, primitive: npt.ArrayLike, start_var_type: PrimitiveIndex, new_values: npt.ArrayLike):
        match self.n_variable_dimensions:
            case 1:
                max_allowed_index = PrimitiveIndex.X_VELOCITY
            case 2:
                max_allowed_index = PrimitiveIndex.Y_VELOCITY 
            case 3:
                max_allowed_index = PrimitiveIndex.Z_VELOCITY 
            case _:
                raise Exception("Unimplemented simulation dimension")
        assert(start_var_type.value <= max_allowed_index.value)
        assert(primitive[start_var_type.value,...].shape == new_values.shape)
        primitive[start_var_type.value,...] = new_values
        return True
    
    def set_velocities(self, primitive: npt.ArrayLike, new_values: npt.ArrayLike):
        assert(primitive[PrimitiveIndex.X_VELOCITY.value:self.last_primitive_velocity_index.value+1,...].shape == new_values.shape)
        primitive[PrimitiveIndex.X_VELOCITY.value:self.last_primitive_velocity_index.value+1,...] = new_values
        return True

    def primitive_to_conservative(self, W: npt.ArrayLike, weight_type: WeightType = WeightType.CENTER ) -> npt.NDArray[np.float64]:
        output = np.zeros(W.shape)
        rho = self.get_primitive_var(W, PrimitiveIndex.DENSITY)
        pressure = self.get_primitive_var(W, PrimitiveIndex.PRESSURE)
        match self.simulation_params.regime:
            case WhichRegime.NEWTONIAN:
                velocities = self.get_velocities(W)                
                velocities_squared = np.power(velocities,2)
                v_mag_2 = np.sum(velocities_squared, axis=0)
                tau = rho* equation_of_state_primitive(self.simulation_params, pressure, rho)+0.5*rho*v_mag_2
                self.set_conservative_var(output,ConservativeIndex.DENSITY,rho)
                self.set_conservative_var(output,ConservativeIndex.TAU, tau)
                first = rho*velocities
                fluxes = first*velocities
                self.set_momentum_fluxes(output, fluxes)
            case WhichRegime.RELATIVITY:
                enthalpy = self.internal_enthalpy_primitive(W, self.simulation_params)
                velocities = self.get_velocities(W)
                alpha  = self.metric.get_metric_product(self.grid_info , self.metric.construct_index(WhichCacheTensor.ALPHA, WeightType.CENTER) ,self.simulation_params) 
                boost = self.metric.W(alpha, velocities, self.grid_info,weight_type, self.simulation_params)
                D = rho*boost
                self.set_conservative_var(output,ConservativeIndex.DENSITY, D)
                self.set_conservative_var(output,ConservativeIndex.TAU, rho*enthalpy*np.power(boost,2)-pressure-D )
                first = rho*enthalpy*np.power(boost,2)
                fluxes = first*velocities
                self.set_momentum_fluxes(output, fluxes)
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
        slices = [slice(1,-1, None)]*(U_cart_padded.ndim)
        exclude_axes = self.excluded_padding_axes(U_cart_padded, VariableSet.CONSERVATIVE)
        for axis in exclude_axes:
            slices[axis] = slice(None) 
        slices = tuple(slices)
        U_cart  = U_cart_padded[slices]
        match self.simulation_params.regime:
            case WhichRegime.NEWTONIAN:
                output = np.zeros(U_cart.shape)
                rho = self.get_conservative_var(U_cart,ConservativeIndex.DENSITY)
                assert(np.all(rho!=0))
                flux = self.get_momentum_fluxes(U_cart)
                velocities = flux/rho
                velocities_squared = np.power(velocities,2)
                v_mag_2 = np.sum(velocities_squared, axis=0)
                E = self.get_conservative_var(U_cart, ConservativeIndex.TAU)
                unclipped_e =  E/rho-0.5*(v_mag_2)
                e = np.clip(unclipped_e, a_min=1E-9, a_max=None)
                assert np.all(e>=0)
                pressure  = equation_of_state_epsilon(self.simulation_params, e,rho )
                flux = self.get_momentum_fluxes(U_cart)
                velocities = flux/rho
                output = np.zeros(U_cart.shape)
                self.set_primitive_var(output, PrimitiveIndex.DENSITY, rho)
                self.set_primitive_var(output, PrimitiveIndex.PRESSURE, pressure)
                self.set_velocities(output, velocities)
                return output
            case WhichRegime.RELATIVITY:
                args = (U_cart, self)
                initial_guess = self.get_primitive_var(self.primitive_previous, PrimitiveIndex.PRESSURE)
                initial_guess = np.maximum(initial_guess, 1e-10)
                recovered_guess, converge, zero_der = newton(self.root_finding_func, initial_guess,args = args, full_output=True)
                assert(np.all(converge))
                out = self.construct_primitives_from_guess(recovered_guess, U_cart)
                return out
            case _:
                raise Exception("Unimplemented relativistic regime")
            
    def root_finding_func(self, guess: npt.ArrayLike,
                                U_cart: npt.ArrayLike, simulation_instance: Self) -> npt.NDArray:
        flux = self.get_momentum_fluxes(U_cart)
        flux_squared = simulation_instance.metric.three_vector_mag_squared(flux, self.grid_info, WeightType.CENTER, simulation_instance.simulation_params)
        D = self.get_conservative_var(U_cart, ConservativeIndex.DENSITY)
        Tau = self.get_conservative_var(U_cart, ConservativeIndex.TAU)
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
        flux = self.get_momentum_fluxes(U_cart)
        flux_squared = self.metric.three_vector_mag_squared(flux, self.grid_info, WeightType.CENTER, self.simulation_params)
        D = self.get_conservative_var(U_cart, ConservativeIndex.DENSITY)
        Tau = self.get_conservative_var(U_cart, ConservativeIndex.TAU)
        z = Tau+guess+D
        z = np.maximum(z, 1e-12)
        v_mag_2 = flux_squared/np.power(z,2)
        v_mag_2 = np.minimum(v_mag_2, 1.0 - 1e-10)
        W2 = np.power(1-v_mag_2,-1)
        W = np.sqrt(W2)
        rho = D/W
        velocities = flux/z
        rho = np.maximum(rho, 1e-12)
        velocities = np.clip(velocities, -0.99999, 0.99999)
        self.set_primitive_var(output, PrimitiveIndex.DENSITY, rho)
        self.set_primitive_var(output, PrimitiveIndex.PRESSURE, guess)
        self.set_velocities(output, velocities)
        return output
        
    def update(self, which_axes : tuple = ()) -> tuple[np.float64, npt.NDArray]:
        # Undo scaling for input
        U_cartesian = self.metric.unweight_system(self.U, self.grid_info, WeightType.CENTER, self.simulation_params)
        dt, state_update_1, primitives = self.LinearUpdate(U_cartesian, which_axes)
        U_1 = self.U+dt*state_update_1
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
        return self.get_primitive_var(W, PrimitiveIndex.DENSITY)*self.get_primitive_var(W,PrimitiveIndex.X_VELOCITY)*np.power(r_center,2)
 

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
            case VariableSet.VECTOR:# NOTE: Hack for now since the only vectors have the variable index at the start
                axis_to_vel_map = {
                    0:0,
                    1: 1,
                    2: 2,
                    3: 3
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
                return [0]
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
                # Add on ghost cells to all edges
                U_padded_cart = self.pad_array(U_cart, self.U_initial_unweighted_padded, VariableSet.CONSERVATIVE)
                W_cart = self.conservative_to_primitive(U_padded_cart)
                W_padded_cart = self.pad_array(W_cart, self.W_initial_unweighted_padded, VariableSet.PRIMITIVE)
                assert(W_padded_cart.shape == U_padded_cart.shape)
                # Figure out which axes to update along (for instance, if doing 1D sim in 3D grid)
                if(len(which_axis)==0):
                    axes = tuple(range(self.n_variable_dimensions))
                else:
                    axes = which_axis
                # Calculate fluxes along all axes
                density_flux_padded = self.density_flux(U_padded_cart, W_padded_cart, WeightType.CENTER)
                tau_flux_padded = self.tau_flux(U_padded_cart, W_padded_cart, WeightType.CENTER)
                momentum_flux_tensor_padded = self.momentum_flux_tensor(U_padded_cart,W_padded_cart, WeightType.CENTER)
                # Calculate quantities that are shared across all axes
                velocities = self.get_velocities(W_cart)
                v_mag_2 = self.metric.three_vector_mag_squared(velocities, self.grid_info, WeightType.CENTER, self.simulation_params)
                alpha = self.metric.get_metric_product(self.grid_info ,self.metric.construct_index(WhichCacheTensor.ALPHA, WeightType.CENTER) ,self.simulation_params) 
                beta = self.metric.get_metric_product(self.grid_info , self.metric.construct_index(WhichCacheTensor.BETA, WeightType.CENTER),self.simulation_params)
                inv_metric =  self.metric.get_metric_product(self.grid_info, self.metric.construct_index(WhichCacheTensor.INVERSE_METRIC, WeightType.CENTER), self.simulation_params)

                v_mag_2_padded = self.pad_array(v_mag_2, self.v_mag2_fixed_padded, VariableSet.VECTOR)
                alpha_padded = self.pad_array( alpha, self.alpha_fixed_padded, VariableSet.SCALAR)
                beta_padded = self.pad_array( beta, self.inverse_beta_fixed_padded, VariableSet.VECTOR)
                inverse_metric_padded = self.pad_array(inv_metric , self.inverse_metric_fixed_padded, VariableSet.METRIC) #  NOTE: Going to treat metric as a scalar. Don't know how matrices should reflect...

                state_update = np.zeros(U_cart.shape)
                pressure = self.get_primitive_var(W_padded_cart, PrimitiveIndex.PRESSURE)
                density = self.get_primitive_var(W_padded_cart, PrimitiveIndex.DENSITY)
                c_s = sound_speed(self.simulation_params,pressure, density)    
                # Calculate fluxes along each axis and update state
                possible_dt = []
                for spatial_index in axes:
                    current_density  =  density_flux_padded[spatial_index,...]
                    target = [1]+[*current_density.shape]
                    current_density = np.reshape(current_density, target)
                    tau  = tau_flux_padded[spatial_index,...]
                    target = [1]+[*tau.shape]
                    current_tau = np.reshape(tau, target)
                    current_mom_flux  = momentum_flux_tensor_padded[spatial_index,:,...]
                    cell_flux = np.concatenate(
                        [
                            current_density,
                            current_tau,
                            current_mom_flux
                        ], 
                        axis=0)
                    # Calculate possible wave speeds along each axis
                    match self.simulation_params.regime:
                        case WhichRegime.NEWTONIAN:
                            lambda_plus = self.get_specific_velocity(W_padded_cart, spatial_index)+c_s
                            lambda_minus = self.get_specific_velocity(W_padded_cart, spatial_index)-c_s
                        case WhichRegime.RELATIVITY:
                            # Eq. 22 of https://iopscience.iop.org/article/10.1086/303604/pdf
                            beta_component = beta_padded[spatial_index,...]
                            gamma_ii = inverse_metric_padded[METRIC_VARIABLE_INDEX.SPACE_1.value+spatial_index,METRIC_VARIABLE_INDEX.SPACE_1.value+spatial_index,...] 
                            velocity_component = self.get_specific_velocity(W_padded_cart, spatial_index)
                            cs_2 = np.power(c_s,2)
                            factor_1 = 1-v_mag_2_padded*cs_2
                            factor_2 = 1-v_mag_2_padded 
                            factor_3 = 1-cs_2
                            prefactor = alpha_padded/factor_1
                            common_factor = velocity_component*factor_3
                            discriminant = factor_2*(gamma_ii*factor_1-np.power(velocity_component,2)*factor_3)
                            lambda_plus = prefactor*(common_factor + c_s*np.sqrt(discriminant)) - beta_component
                            lambda_minus = prefactor*(common_factor - c_s*np.sqrt(discriminant)) - beta_component
                    alpha_minus, alpha_plus = self.alpha_plus_minus(lambda_plus, lambda_minus, spatial_index)
                    alpha_sum = alpha_minus+alpha_plus
                    assert(np.all(alpha_sum != 0))
                    excluded_axes = self.excluded_padding_axes(U_padded_cart, VariableSet.CONSERVATIVE)
                    alpha_prod = alpha_minus*alpha_plus

                    # Construct left and right states at interfaces
                    slices_left = [slice(None)]*cell_flux.ndim
                    for axis in excluded_axes:
                        slices_left[axis] = slice(None,None,None) # Select all variables
                    # NOTE: Assuming that excluded axes are always the first axes for Conservative Variable set
                    slices_left[spatial_index+1] = slice(0,-1,None)
                    slices_left = tuple(slices_left)
                    slices_right = [slice(None)]*cell_flux.ndim
                    for axis in excluded_axes:
                        slices_right[axis] = slice(None,None,None) 
                    slices_right[spatial_index+1] = slice(1,None,None)
                    slices_right = tuple(slices_right)

                    cell_flux_left = cell_flux[slices_left]
                    conserve_left = U_padded_cart[slices_left]
                    cell_flux_right = cell_flux[slices_right]
                    conserve_right = U_padded_cart[slices_right]
                    flux_interface = (alpha_plus * cell_flux_left + alpha_minus * cell_flux_right
                                - alpha_prod * (conserve_right - conserve_left)) / (alpha_sum)
                    remove_ghost_slices = [slice(None)]*flux_interface.ndim
                    for axis in range(flux_interface.ndim):
                        if axis in excluded_axes:
                            continue 
                        if axis != spatial_index+1:
                            remove_ghost_slices[axis] = slice(1,-1,None)
                    flux_interface_unpadded = flux_interface[tuple(remove_ghost_slices)]
            #       flux_densitized = self.metric.weight_system(flux_interface, self.grid_info, WeightType.EDGE, self.simulation_params)
                    axis_weights = self.n_variable_dimensions*[WeightType.CENTER]
                    axis_weights[spatial_index] = WeightType.EDGE
                    weights = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.DETERMINANT, tuple(axis_weights)), self.simulation_params)
                    flux_densitized = flux_interface_unpadded * weights
                    flux_interface_left = flux_densitized[slices_left]
                    flux_interface_right = flux_densitized[slices_right]
                    flux_change = (flux_interface_right - flux_interface_left)/self.grid_info.delta(spatial_index)[spatial_index]
                # NOTE: Possible bug here with dt calculation. Only largest possible dt from all axes should be taken
                    possible_dt.append(self.calc_dt(alpha_plus, alpha_minus))
                    state_update += flux_change
                dt = max(possible_dt)
                exclude_axes = self.excluded_padding_axes(W_padded_cart, VariableSet.PRIMITIVE)
                slices = [slice(1,-1, None)]*(W_padded_cart.ndim) # Remove ghost cells from padded grid
                for axis in exclude_axes:
                    slices[axis] = slice(None)
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
        return three_velocities_padded - shift_vec_padded

    def density_flux(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index)
        velocities = self.get_velocities(W_padded_cart)
        shifted_velocity = self.shift_three_velocity_padded(velocities)
        D = self.get_conservative_var(U_padded_cart, ConservativeIndex.DENSITY)
        return D*shifted_velocity

    def momentum_flux_tensor(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index, spatial_index) where the first indexes the coordinate and the 2nd indexes the direction
        velocities = self.get_velocities(W_padded_cart)
        shifted_velocity = self.shift_three_velocity_padded(velocities)
        pressure = self.get_primitive_var(W_padded_cart,PrimitiveIndex.PRESSURE)
        flux = self.get_momentum_fluxes(U_padded_cart)
        pressure_tensor = pressure*self.dirac_delta_constant
        KE = np.einsum("i...,j...->ij...", flux, shifted_velocity)
        return KE+pressure_tensor
    
    def tau_flux(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index)
        Tau =  self.get_conservative_var(U_padded_cart, ConservativeIndex.TAU)
        pressure = self.get_primitive_var(W_padded_cart,PrimitiveIndex.PRESSURE)
        velocities = self.get_velocities(W_padded_cart)
        shifted_velocities = self.shift_three_velocity_padded(velocities)
        return (Tau*shifted_velocities+ pressure*velocities)

    def alpha_plus_minus(self, lambda_plus: npt.ArrayLike, lambda_minus: npt.ArrayLike, spatial_index:int ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        alpha_plus = np.zeros(lambda_plus.shape)
        alpha_minus = np.zeros(lambda_minus.shape)
        assert(alpha_minus.shape == alpha_plus.shape)
        # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for + and MAX(0,-left, -right)
        # Grab the left and right speeds from the padded array
        everything = [slice(None,None,None)]*alpha_plus.ndim
        slices_left = everything.copy()
        slices_right = everything.copy()
        slices_left[spatial_index] = slice(0,-1,None)
        slices_right[spatial_index] = slice(1,None,None)
        slices_left = tuple(slices_left)
        slices_right = tuple(slices_right)
        lambda_plus_left = lambda_plus[slices_left]
        lambda_plus_right = lambda_plus[slices_right]
        lambda_minus_left = lambda_minus[slices_left]
        lambda_minus_right = lambda_minus[slices_right]
        assert(lambda_plus_left.shape == lambda_plus_right.shape == lambda_minus_left.shape == lambda_minus_right.shape)
        zeros = np.zeros(lambda_plus_left.shape)
        # First, the plus case
        alpha_plus = np.maximum(zeros , np.maximum(lambda_plus_left ,lambda_plus_right ))
        alpha_minus = np.maximum(zeros , np.maximum(-lambda_minus_left ,-lambda_minus_right ))
        return (alpha_minus, alpha_plus)
    
    def StressEnergyTensor(self,W_cart_unpadded:npt.ArrayLike):
        # TODO: Change 4 vector to include normalization factor (Did for SR, need to generalize to GR)
        metric = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.METRIC, WeightType.CENTER), self.simulation_params, use_cache=True)
        rho  = self.get_primitive_var(W_cart_unpadded, PrimitiveIndex.DENSITY)
        pressure  = self.get_primitive_var(W_cart_unpadded, PrimitiveIndex.PRESSURE)
        velocities = self.get_velocities(W_cart_unpadded) # spatial components of 4 velocity
        four_vel_shape  = [*velocities.shape]
        four_vel_shape[-1]  = four_vel_shape[-1]+1 # Add one for 0 component
        four_velocities  = np.zeros (four_vel_shape)
        four_velocities[0,...] = 1 # 0th component
        # First shift the spatial velocities to the current spacelike slice
        shifted_three_velocities = self.metric.shift_three_velocity(velocities, self.grid_info, self.simulation_params)
        four_velocities[1:,...] = shifted_three_velocities 
        # Scale by the boost
        alpha = self.metric.get_metric_product(self.grid_info, self.metric.construct_index(WhichCacheTensor.ALPHA, WeightType.CENTER),self.simulation_params)
        W = self.metric.W(alpha, velocities, self.grid_info,WeightType.CENTER, self.simulation_params)
        four_velocities[...] = W*four_velocities[...]
        u_u  = np.zeros(metric.shape) # Shape of (grid_size, first, secon)
        # Help from Gemini . Prompt:  I have a numpy array of shape (10, 10, 2). I want to take the outer product along the last axis and end up with an array of shape (10,10,2,2) . Asked for generalization for einsum
        # What it does: Takes the outer product on the last index, then moves the two indices to the front (to be compatible with the ordering of the metric field)
        u_u = np.einsum('...k,...l->kl...', four_velocities, four_velocities)
        inv_metric = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.INVERSE_METRIC, WeightType.CENTER), self.simulation_params, use_cache=True)
        t_mu_nu_raised = rho*self.internal_enthalpy_primitive(W_cart_unpadded, self.simulation_params)*u_u +pressure*inv_metric
        assert(0)
        return t_mu_nu_raised

    def SourceTerm(self,W:npt.ArrayLike):
        # Assumes that W is the array of Cartesian primitive variables. Needs to be unpadded due to construct_grid_centers call
        # primitive variables 
        # W = (\rho, P, v_{j})

        output = np.zeros(W.shape)
        # Hack to make Bondi work without implementing Source Term w.r.t. Metric and stress energy
        # gri  =self.grid_info.mesh_grid(WeightType.CENTER)
        # rho  = self.get_primitive_var(W, PrimitiveIndex.DENSITY)
        # pressure  = self.get_primitive_var(W, PrimitiveIndex.PRESSURE)
        # self.set_conservative_var(output, ConservativeIndex.DENSITY, 0)
        # self.set_conservative_var(output, ConservativeIndex.TAU, -rho*self.get_primitive_var(W,PrimitiveIndex.X_VELOCITY)*self.simulation_params.GM)
        # self.set_conservative_var(output, ConservativeIndex.X_MOMENTUM_DENSITY, 2*gri[0]*pressure - rho*self.simulation_params.GM)
        # self.set_conservative_var(output, ConservativeIndex.Y_MOMENTUM_DENSITY, 0)
        # self.set_conservative_var(output, ConservativeIndex.Z_MOMENTUM_DENSITY, 0)
        # return output
        T_mu_nu_raised = self.StressEnergyTensor(W)
        metric = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.METRIC, WeightType.CENTER), self.simulation_params)
        partial_metric = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.PARTIAL_DER, WeightType.CENTER), self.simulation_params)
        partial_ln_alpha = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.PARTIAL_LN_ALPHA, WeightType.CENTER), self.simulation_params)
        alpha =self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.ALPHA, WeightType.CENTER), self.simulation_params)
        Christoffel_upper = self.metric.get_metric_product(self.grid_info,  self.metric.construct_index(WhichCacheTensor.CHRISTOFFEL_UPPER0, WeightType.CENTER), self.simulation_params)
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
        self.set_conservative_var(output,ConservativeIndex.DENSITY, rho_source)
        self.set_conservative_var(output,ConservativeIndex.TAU, energy_source)
        self.set_momentum_fluxes(output, source_flux[METRIC_VARIABLE_INDEX.SPACE_1.value:,...])
        return output

    def calc_dt(self, alpha_plus: npt.ArrayLike, alpha_minus:npt.ArrayLike):
        max_alpha = np.max( [alpha_plus, alpha_minus]) 
        deltas = np.asarray([np.min(self.grid_info.delta(i)) for i in range(self.n_variable_dimensions)])
        min_delta = np.min(deltas[deltas>0])
        return self.simulation_params.Courant*min_delta/max_alpha

    def internal_enthalpy_primitive(self,W: npt.ArrayLike, sim_params: SimParams) -> npt.NDArray[np.float64]:
        pressure = self.get_primitive_var(W, PrimitiveIndex.PRESSURE)
        density = self.get_primitive_var(W, PrimitiveIndex.DENSITY)
        return internal_enthalpy_primitive_raws(pressure, density, sim_params)

if __name__ == "__main__":
    pass
