import numpy.typing as npt
from enum import Enum
import numpy as np
from numpy.lib._index_tricks_impl import ndindex
from numpy.lib._arraypad_impl import _as_pairs
from GridInfo import GridInfo, WeightType
from UpdateSteps import TimeUpdateType,SpatialUpdate, SpatialUpdateType
from BoundaryManager import BoundaryConditionManager
from metrics import Metric
from metrics.Metric import WhichCacheTensor, METRIC_VARIABLE_INDEX
from HelperFunctions import *
from GuessPrimitives import *
from EquationOfState import *
from scipy.optimize import newton

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
        self.simulation_params = sim_params
        self.primitive_previous = primitive_tensor
        self.grid_info = a_grid_info 
        self.bcm = a_bcm
        self.metric = a_metric
        U_unweighted_initial = self.primitive_to_conservative(primitive_tensor) 
        padding = self.simulation_params.spatial_integration.pad_width()
        pad_width = [(padding,padding)]*(U_unweighted_initial.ndim-1)+ [(0,0)]
        self.U_initial_unweighted_padded = np.pad(U_unweighted_initial, pad_width, initial_value_boundary_padding) 
        self.W_initial_unweighted_padded = np.pad(primitive_tensor, pad_width, initial_value_boundary_padding) 
        assert(self.W_initial_unweighted_padded.shape == self.U_initial_unweighted_padded.shape)
        self.U = self.metric.weight_system(U_unweighted_initial,  self.grid_info, WeightType.CENTER, self.simulation_params)
        self.current_time = starting_time
        # Purely spatial dirac delta field for flux tensor calculation
        spatial_dims = [self.n_variable_dimensions]*2+[*self.U_initial_unweighted_padded[...,0].shape]
        self.dirac_delta_constant = np.zeros(spatial_dims) ## (grisize, spatial, spatial)
        for i in range(self.n_variable_dimensions):
            self.dirac_delta_constant [i,i,...] = 1

           
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
                velocities = W[...,PrimitiveIndex.X_VELOCITY.value:] # Assuming velocities are the trailing variables are part of 4 velocity. Size of (gridsize, dim)
                alpha  = self.metric.get_metric_product(self.grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, self.simulation_params) 
                boost = self.metric.boost_field_four_vel(alpha, velocities, self.grid_info,weight_type, self.simulation_params)
                D = rho*boost
                output[...,ConservativeIndex.DENSITY.value] = D
                output[...,ConservativeIndex.TAU.value] = rho*enthalpy*np.power(boost,2)-pressure-D 
                first = rho*enthalpy*np.power(boost,1) # Not W^2 since velocity is 4 velocity, so you get rid of an extra boost
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
                rho = index_conservative_var(U_cart_padded,ConservativeIndex.DENSITY, self.n_variable_dimensions)
                assert(np.all(rho!=0))
                velocities = (U_cart_padded[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:].T/rho).T
                velocities_squared = np.power(velocities,2)
                v_mag_2 = np.sum(velocities_squared, axis=-1).T
                E = index_conservative_var(U_cart_padded, ConservativeIndex.TAU, self.n_variable_dimensions)
                unclipped_e =  E/rho-0.5*v_mag_2              
                e = np.clip(unclipped_e, a_min=1E-9, a_max=None)
                assert np.all(e>=0)
                pressure  = pressure_from_epsilon(self.simulation_params, e,rho )
                velocities = (U_cart_padded[...,ConservativeIndex.X_MOMENTUM_DENSITY.value:].T/rho).T
                output = np.zeros(U_cart_padded.shape)
                output[...,PrimitiveIndex.DENSITY.value] = rho
                output[..., PrimitiveIndex.X_VELOCITY.value:] = velocities
                output[...,PrimitiveIndex.PRESSURE.value] =  pressure
                return output
            case WhichRegime.RELATIVITY:
                args = (U_cart_padded, self.metric, self.simulation_params, self.grid_info, self.n_variable_dimensions)
                initial_guess = index_primitive_var(self.primitive_previous, PrimitiveIndex.PRESSURE, self.n_variable_dimensions)
                # print("Init",initial_guess)
                # print("Init",log_pressure_guess)
                recovered_pressure_guess = newton(pressure_finding_func, initial_guess,args = args, fprime=pressure_finding_func_der, maxiter=5)
                out = construct_primitives_from_guess(recovered_pressure_guess, U_cart_padded, self.metric, self.simulation_params, self.grid_info, self.n_variable_dimensions)
#                assert((index_primitive_var(out, PrimitiveIndex.DENSITY, self.n_variable_dimensions)>=0).all())
                return out
            case _:
                raise Exception("Unimplemented relativistic regime")
        
    def update(self, which_axes : tuple = ()) -> tuple[np.float64, npt.NDArray]:
        # Undo scaling for input
        U_cartesian = self.metric.unweight_system(self.U, self.grid_info, WeightType.CENTER, self.simulation_params)
        dt, state_update_1, primitives = self.LinearUpdate(U_cartesian, which_axes)
        U_1 = self.U+dt*state_update_1
        assert((index_primitive_var(primitives, PrimitiveIndex.PRESSURE, self.n_variable_dimensions)>=0).all())
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
 
    def pad_unweighted_array(self, var:npt.ArrayLike,which_var: WhichVar):
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
        match which_var:
            case WhichVar.CONSERVATIVE:
                assert(self.U_initial_unweighted_padded.shape == padded.shape)
            case WhichVar.PRIMITIVE:
                assert(self.W_initial_unweighted_padded.shape == padded.shape)
            case _:
                raise Exception("Unimplemented variable type")

        # And apply along each axis

        for axis in range(0,padded.ndim):
            # Iterate using ndindex as in apply_along_axis, but assuming that
            # function operates inplace on the padded array.

            # view with the iteration axis at the end
            view = np.moveaxis(padded, axis, -1)
            match which_var:
                case WhichVar.CONSERVATIVE:
                    initial_view = np.moveaxis(self.U_initial_unweighted_padded,axis, -1)
                case WhichVar.PRIMITIVE:
                    initial_view = np.moveaxis(self.W_initial_unweighted_padded,axis, -1)
                case _:
                    raise Exception("Unimplemented variable type")


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

    def LinearUpdate(self, U_cart: npt.ArrayLike, which_axis: tuple = ()): 
        # Returns the time step needed, the change in the conservative variables, and the recovered primitives of the system prior to stepping
        match self.simulation_params.spatial_integration.method:
            case SpatialUpdateType.FLAT:
                U_padded_cart = self.pad_unweighted_array(U_cart, WhichVar.CONSERVATIVE)
                W_cart = self.conservative_to_primitive(U_padded_cart)
                W_padded_cart = self.pad_unweighted_array(W_cart, WhichVar.PRIMITIVE)
                assert(W_padded_cart.shape == U_padded_cart.shape)
                possible_dt = []
                state_update = np.zeros(U_cart.shape)
                if(len(which_axis)==0):
                    axes = tuple(range(self.n_variable_dimensions))
                else:
                    axes = which_axis
                density_flux = self.density_flux(U_padded_cart, W_padded_cart, WeightType.CENTER)
                tau_flux = self.tau_flux(U_padded_cart, W_padded_cart, WeightType.CENTER)
                momentum_flux_tensor = self.momentum_flux_tensor(U_padded_cart,W_padded_cart, WeightType.CENTER)
                for dim in axes:
                    flux_change, alpha_plus, alpha_minus = self.spatial_derivative(U_padded_cart,W_padded_cart,
                                                                                   density_flux, momentum_flux_tensor, tau_flux,
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
    
    def density_flux(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index)
        D = U_padded_cart[...,ConservativeIndex.DENSITY.value]
        velocity = W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value:]
        return (D.T*velocity.T).T

    def momentum_flux_tensor(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index, spatial_index) where the first indexes the coordinate and the 2nd indexes the direction
        density = W_padded_cart[...,PrimitiveIndex.DENSITY.value]
        velocity = W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value:]
        pressure = index_primitive_var(W_padded_cart,PrimitiveIndex.PRESSURE,self.n_variable_dimensions)
        pressure_tensor = pressure*self.dirac_delta_constant
        KE = density*np.einsum("...i,...j->ij...", velocity, velocity)
        output = np.einsum("ij...->...ij", KE+pressure_tensor)
        return output
    
    def tau_flux(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, weight_type: WeightType) -> npt.NDArray:
        # (gridsize, spatial_index)
        Energy =  index_conservative_var(U_padded_cart, ConservativeIndex.TAU, self.n_variable_dimensions)
        pressure = index_primitive_var(W_padded_cart,PrimitiveIndex.PRESSURE,self.n_variable_dimensions)
        velocity = W_padded_cart[...,PrimitiveIndex.X_VELOCITY.value:]
        return ((Energy+pressure).T*velocity.T).T

    def spatial_derivative(self, U_padded_cart: npt.ArrayLike, W_padded_cart: npt.ArrayLike, 
                           density_flux: npt.ArrayLike, momentum_flux_tensor: npt.ArrayLike, tau_flux: npt.ArrayLike,
                             spatial_index: np.uint = 0) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # Assuming that U is Cartesian. Cell_scaling is for the fluxes
        zero  =  density_flux[...,spatial_index]
        target = [*zero.shape]+[1]
        zero = zero.reshape(target)
        first  = tau_flux[...,spatial_index]
        first = first.reshape(target)
        rest  = momentum_flux_tensor[...,spatial_index,:]
        cell_flux = np.concatenate(
            [
                zero,
                first,
                rest
            ], 
            axis=-1)
        alpha_minus, alpha_plus = self.alpha_plus_minus(W_padded_cart)
        alpha_sum = alpha_minus+alpha_plus 
        assert(np.all(alpha_sum != 0))
        alpha_prod = alpha_minus*alpha_plus
        # Bunch of .T because numpy broadcasting rules
        slices_left_plus = tuple([slice(1,-1,None)]*self.n_variable_dimensions)
        slices_right_plus = tuple([slice(2,None,None)]*self.n_variable_dimensions)
        slices_left_minus = tuple([slice(None,-2,None)]*self.n_variable_dimensions)
        slices_right_minus = tuple([slice(1,-1,None)]*self.n_variable_dimensions)
        left_cell_flux_plus = cell_flux[slices_left_plus].T 
        left_conserve_plus = U_padded_cart[slices_left_plus].T
        right_cell_flux_plus =cell_flux[slices_right_plus].T  
        right_conserve_plus = U_padded_cart[slices_right_plus].T
        left_cell_flux_minus = cell_flux[slices_left_minus].T 
        left_conserve_minus = U_padded_cart[slices_left_minus].T
        right_cell_flux_minus =cell_flux[slices_right_minus].T  
        right_conserve_minus = U_padded_cart[slices_right_minus].T
        cell_flux_plus_half = (alpha_plus.T*left_cell_flux_plus+ alpha_minus.T*right_cell_flux_plus
                               -alpha_prod.T*(right_conserve_plus -left_conserve_plus))/alpha_sum.T
        cell_flux_minus_half = (alpha_plus.T*left_cell_flux_minus+ alpha_minus.T*right_cell_flux_minus
                                -alpha_prod.T*(right_conserve_minus-left_conserve_minus))/alpha_sum.T
        weights = self.metric.cell_weights(self.grid_info, WeightType.EDGE, self.simulation_params) 
        slices_plus_half = [slice(1,None, None)]*weights.ndim
        slices_plus_half = tuple(slices_plus_half)
        slices_minus_half = [slice(None,-1, None)]*weights.ndim
        slices_minus_half = tuple(slices_minus_half)
        cell_flux_plus_half_rescaled = cell_flux_plus_half* weights[slices_plus_half].T
        cell_flux_minus_half_rescaled = cell_flux_minus_half* weights[slices_minus_half].T
        # cell_flux_plus_half_rescaled = cell_flux_plus_half* weights[1:,...]
        # cell_flux_minus_half_rescaled = cell_flux_minus_half* weights[:-1,...]         
        return -(cell_flux_plus_half_rescaled.T-cell_flux_minus_half_rescaled.T)/self.grid_info.delta(spatial_index)[spatial_index], alpha_plus, alpha_minus

    def alpha_plus_minus(self, primitives: npt.ArrayLike) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pressure = index_primitive_var(primitives, PrimitiveIndex.PRESSURE,self.n_variable_dimensions)
        density = index_primitive_var(primitives, PrimitiveIndex.DENSITY,self.n_variable_dimensions)
        c_s = sound_speed(self.simulation_params,pressure, density)
        lambda_plus = primitives[...,PrimitiveIndex.X_VELOCITY.value]+c_s
        lambda_minus = primitives[...,PrimitiveIndex.X_VELOCITY.value]-c_s
        # We have two lists of sounds speeds in the left and right cells. Do MAX(0, left, right) for + and MAX(0,-left, -right) for - 
        zeros_shape = [dim-2 for dim in lambda_plus.shape]
        zeros = np.zeros(zeros_shape)
        # Grab the left and right speeds from the padded array
        slices_left = tuple([slice(None,-2,None)]*self.n_variable_dimensions)
        slices_right = tuple([slice(2,None,None)]*self.n_variable_dimensions)
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
        velocities = W_cart_unpadded[...,2:] # spatial components of 4 velocity
        four_vel_shape  = [*velocities.shape]
        four_vel_shape[-1]  = four_vel_shape[-1]+1 # Add one for 0 component
        four_velocities  = np.zeros (four_vel_shape)
        four_velocities[...,1:]  = velocities # fill in spatial components already
        alpha  = self.metric.get_metric_product(self.grid_info , WhichCacheTensor.ALPHA,  WeightType.CENTER, self.simulation_params) 
        boost = self.metric.boost_field_four_vel(alpha, velocities, self.grid_info,WeightType.CENTER, self.simulation_params)
        four_velocities[...,0] = boost # 0th component
        u_u  = np.zeros(metric.shape) # Shape of (grid_size, first, secon)
        # Help from Gemini . Prompt:  I have a numpy array of shape (10, 10, 2). I want to take the outer product along the last axis and end up with an array of shape (10,10,2,2) . Asked for generalization for einsum
        # What it does: Takes the outer product on the last index, then moves the two indices to the front (to be compatible with the ordering of the metric field)
        u_u = np.einsum('...k,...l->kl...', four_velocities, four_velocities)
        t_mu_nu_raised = rho*internal_enthalpy_primitive(W_cart_unpadded, self.simulation_params, self.n_variable_dimensions)*u_u +pressure*metric
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
