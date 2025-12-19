import numpy as np 
import numpy.typing as npt
from GridInfo import WeightType, GridInfo
from typing import Tuple, List
from metrics.Metric import Metric, CachedTensor, METRIC_VARIABLE_INDEX
from CommonClasses import SimParams

class CartesianMinkowski_1_2(Metric):
    dimension: np.float64  = None 
    cached_metric_center:  CachedTensor  = CachedTensor(None, WeightType.CENTER)
    cached_inv_metric_center:  CachedTensor  = CachedTensor(None, WeightType.CENTER)
    cached_determinant_center:  CachedTensor  = CachedTensor(None, WeightType.CENTER)
    cached_determinant_edge: CachedTensor = CachedTensor(None, WeightType.EDGE)
    cached_partial_der_metric_center: CachedTensor  = CachedTensor(None, WeightType.CENTER)
    cached_christoffel_upper0_center: CachedTensor  = CachedTensor(None, WeightType.CENTER)

    def __init__(self, grid_info: GridInfo, sim_params: SimParams):
        super().__init__()
        self.dimension = 3
        self.sanity_check(grid_info, sim_params)

    def metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        output[METRIC_VARIABLE_INDEX.TIME.value, METRIC_VARIABLE_INDEX.TIME.value,...] = -1
        output[METRIC_VARIABLE_INDEX.SPACE_1.value, METRIC_VARIABLE_INDEX.SPACE_1.value,...] = 1
        output[METRIC_VARIABLE_INDEX.SPACE_2.value, METRIC_VARIABLE_INDEX.SPACE_2.value,...] = 1
        return output 
    
    def inv_metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        return self.metric(mesh_grid, expected_product_size, sim_params)

    def determinant(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) -> npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        output[...] = -1
        return output 
    
    def partial_derivative(self, mesh_grid: Tuple[npt.NDArray[np.float64],...] , expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        return np.zeros(expected_product_size)

    def Christoffel_upper(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        return np.zeros(expected_product_size)
    
    def partial_ln_alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] , sim_params: SimParams) ->  npt.NDArray[np.float64]:
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