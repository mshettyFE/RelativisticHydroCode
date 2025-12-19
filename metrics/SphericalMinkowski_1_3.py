import numpy as np 
import numpy.typing as npt
from GridInfo import WeightType, GridInfo
from typing import Tuple, List
from metrics.Metric import Metric, cached_array, METRIC_VARIABLE_INDEX
from CommonClasses import SimParams

class SphericalMinkowski_1_3(Metric):
    dimension: np.float64  = None 
    cached_metric_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_inv_metric_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_determinant_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_determinant_edge: cached_array = cached_array(None, WeightType.EDGE)
    cached_partial_der_metric_center: cached_array  = cached_array(None, WeightType.CENTER)
    cached_christoffel_upper0_center: cached_array  = cached_array(None, WeightType.CENTER)

    def __init__(self, grid_info: GridInfo, sim_params: SimParams):
        super().__init__()
        self.dimension = 4
        self.sanity_check(grid_info, sim_params)

    def metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        # ds^{2} = -dt^{2}+dr^2 + r^2d\theta^2+r^2 \sin^{2}\theta d\phi^{2}
        output = np.zeros(expected_product_size)
        output[METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value,...] = -1
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = 1
        r_2  =np.power(mesh_grid[0],2) # r^2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = r_2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = r_2* np.power( np.sin(mesh_grid[1]),2) # r^2\sin^{\theta}
        return output
    
    def inv_metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        output[METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value,...] = -1
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = 1
        r_minus_2  =np.power(mesh_grid[0],-2) # r^2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = r_minus_2
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = r_minus_2* np.power( np.sin(mesh_grid[1]),-2) # 1/(r^2\sin^2(\theta))
        return output 

    def determinant(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) -> npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        output[...] =-1* np.power(mesh_grid[0],4)* np.power( np.sin(mesh_grid[1]),2) 
        return output 
    
    def partial_derivative(self, mesh_grid: Tuple[npt.NDArray[np.float64],...] , expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        # Only R (SPACE_1) an theta (Space_2) have partial derivatives
        output  =np.zeros(expected_product_size)
        output[METRIC_VARIABLE_INDEX.SPACE_1.value,METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...]  =  2*mesh_grid[0] #2r
        output[METRIC_VARIABLE_INDEX.SPACE_1.value,METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...]  =  2*mesh_grid[0]*np.power(np.sin(mesh_grid[1]),2) #2r \sin^{2}\theta
        output[METRIC_VARIABLE_INDEX.SPACE_2.value,METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...]  =  2*np.power(mesh_grid[0],2)*np.sin(mesh_grid[1])*np.cos(mesh_grid[1]) #2r^{2} \sin\theta \cos \theta
        return output

    def Christoffel_upper(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        # Taken from https://math.stackexchange.com/questions/1985964/christoffel-symbols-for-spherical-polar-coordinates
        # R
        output  =np.zeros(expected_product_size)
        output[METRIC_VARIABLE_INDEX.SPACE_1.value,METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...]  =  -mesh_grid[0] #2r
        output[METRIC_VARIABLE_INDEX.SPACE_1.value,METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...]  =  -mesh_grid[0]*np.power(np.sin(mesh_grid[1]),2) #2r        
        # \theta
        r_minus_1  = np.power(mesh_grid[0],-1)
        output[METRIC_VARIABLE_INDEX.SPACE_2.value,METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...]  =  r_minus_1
        output[METRIC_VARIABLE_INDEX.SPACE_2.value,METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...]  =  r_minus_1
        output[METRIC_VARIABLE_INDEX.SPACE_2.value,METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...]  =  -np.sin(mesh_grid[1])*(np.cos(mesh_grid[1]))
        #\phi
        output[METRIC_VARIABLE_INDEX.SPACE_3.value,METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...]  =  r_minus_1
        output[METRIC_VARIABLE_INDEX.SPACE_3.value,METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...]  =  r_minus_1
        cot = np.power(np.tan(mesh_grid[1]),-1)
        output[METRIC_VARIABLE_INDEX.SPACE_3.value,METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_3.value,...]  = cot
        output[METRIC_VARIABLE_INDEX.SPACE_3.value,METRIC_VARIABLE_INDEX.SPACE_3.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...]  =  cot
        return output
    
    def partial_ln_alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) ->  npt.NDArray[np.float64]:
        # Fix upper index to 0 since that's the only one that's relevant for this problem
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output     
    
    def alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] , sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Fix upper index to 0 since that's the only one that's relevant for this problem
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.ones(expected_product_size)
        return output 
    
    def beta(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output