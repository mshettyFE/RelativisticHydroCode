import numpy as np 
import numpy.typing as npt
from GridInfo import WeightType, GridInfo
from typing import Tuple, List
from metrics.Metric import Metric, cached_array, METRIC_VARIABLE_INDEX
from HelperFunctions import SimParams


class LinearGravity_1_3(Metric):
    dimension: np.float64  = None 
    cached_metric_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_metric_edge: cached_array  = cached_array(None, WeightType.EDGE)
    cached_inv_metric_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_inv_metric_edge: cached_array  = cached_array(None, WeightType.EDGE)
    cached_determinant_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_determinant_edge: cached_array = cached_array(None, WeightType.EDGE)
    cached_partial_der_metric_center: cached_array  = cached_array(None, WeightType.CENTER)
    cached_christoffel_upper0_center: cached_array  = cached_array(None, WeightType.CENTER)

    def __init__(self, grid_info: GridInfo, sim_params: SimParams):
        super().__init__()
        self.dimension = 4
        self.sanity_check(grid_info, sim_params)

    def phi(self,  mesh_grid: Tuple[npt.NDArray[np.float64],...],  sim_params: SimParams):
        r = mesh_grid[0]
        assert(r.all() != 0)
        return -sim_params.GM/r
    
    def dPhi_dr(self,  mesh_grid: Tuple[npt.NDArray[np.float64],...],  sim_params: SimParams):
        r = mesh_grid[0]
        return  sim_params.GM*np.power(r,-2)
    
    def metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        # https://profoundphysics.com/christoffel-symbols-a-complete-guide-with-examples/ under Christoffel Symbols For The Weak-Field Metric
        output = np.zeros(expected_product_size)
        phi = self.phi(mesh_grid, sim_params)
        output[METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value,...] = -(1+2*phi)
        scale  = (1-2*phi)
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = scale
        r_2  =np.power(mesh_grid[0],2) # r^2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = scale*r_2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = scale*r_2* np.power( np.sin(mesh_grid[1]),2) 
        return output 
    
    def inv_metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        phi = self.phi(mesh_grid, sim_params)
        output[METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value,...] = np.power(-(1+2*phi),-1)
        scale  =np.power((1-2*phi),-1)
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = scale
        r_minus_2  =np.power(mesh_grid[0],-2) # r^2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = scale*r_minus_2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = scale*r_minus_2* np.power( np.sin(mesh_grid[1]),-2) 
        return output 

    def determinant(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) -> npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        phi = self.phi(mesh_grid, sim_params)
        r = mesh_grid[0]
        output[...] = -(1+2*phi)*np.power((1-2*phi) ,3) * np.power(r,4) * np.power( np.sin(mesh_grid[1]),2) 
        return output 
    
    def partial_derivative(self, mesh_grid: Tuple[npt.NDArray[np.float64],...] , expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # https://profoundphysics.com/christoffel-symbols-a-complete-guide-with-examples/ under Christoffel Symbols For The Weak-Field Metric
        # TODO: Get this  done
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output  =np.zeros(expected_product_size)
        r = mesh_grid[0]
        dPhi_dr = self.dPhi_dr(mesh_grid, sim_params)
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value,...] = -2*dPhi_dr
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = -2*dPhi_dr
        prefactor = 2*(sim_params.GM+r)
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = prefactor
        theta= mesh_grid[1]
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...] = prefactor*np.power(np.sin(theta),2)
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...] = np.sin(theta)*np.cos(theta)*(1-2*self.phi(mesh_grid,sim_params))*np.power(r,2)
        return output

    def Christoffel_upper(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] , sim_params: SimParams) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # https://profoundphysics.com/christoffel-symbols-a-complete-guide-with-examples/ under Christoffel Symbols For The Weak-Field Metric
        # TODO: Get this  done
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output  =np.zeros(expected_product_size)
        r = mesh_grid[0]
        dPhi_dr = self.dPhi_dr(mesh_grid, sim_params)
        # time
        output[METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = dPhi_dr
        output[METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.TIME.value,...] = dPhi_dr
        # r
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value,...] = dPhi_dr
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = -dPhi_dr
        prefactor = (r*dPhi_dr-1)*r
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = prefactor
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...] = prefactor*np.power(np.sin(mesh_grid[0]),2)
        # theta
        prefactor = (1-r*dPhi_dr)/r
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = prefactor
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = prefactor
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...] = -np.sin(mesh_grid[0])*np.cos(mesh_grid[1])
        # phi
        output[METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...] = prefactor
        output[METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = prefactor
        output[METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...] = np.power(np.tan(mesh_grid[1]),-1)
        output[METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = np.power(np.tan(mesh_grid[1]),-1)
        return output
    
    def partial_ln_alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) ->  npt.NDArray[np.float64]:
        # Fix upper index to 0 since that's the only one that's relevant for this problem
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        phi = self.phi(mesh_grid, sim_params)
        output[...]  = np.log(np.sqrt(1+2*phi))
        return output
    
    def alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] , sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Fix upper index to 0 since that's the only one that's relevant for this problem
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        phi = self.phi(mesh_grid, sim_params)
        output[...]  = np.sqrt(1+2*phi)
        return output 