import numpy as np 
from abc import ABC,abstractmethod 
from dataclasses import dataclass
import numpy.typing as npt
from GridInfo import WeightType, GridInfo
from enum import Enum
from CommonClasses import SimParams
from typing import Tuple, Dict

class METRIC_VARIABLE_INDEX(Enum):
    TIME = 0
    SPACE_1 = 1
    SPACE_2 = 2
    SPACE_3 = 3

class WhichCacheTensor(Enum):
    METRIC =  0
    INVERSE_METRIC  =  1
    DETERMINANT =  2
    PARTIAL_DER = 3
    CHRISTOFFEL_UPPER0  = 4
    PARTIAL_LN_ALPHA = 5
    ALPHA  = 6
    BETA = 7

TensorIndex = Tuple[WhichCacheTensor, Tuple[WeightType,...]]
MeshGrid = npt.NDArray[np.float64]
TensorData = npt.NDArray[np.float64]

@dataclass
class CachedTensor:
    array: TensorData
    mesh_grid: MeshGrid

class Metric(ABC):
    dimension: np.float64  = None 
    cached_tensor_data: Dict[TensorIndex, CachedTensor] = {}

    def validate_index(self, index: TensorIndex) -> bool:
        assert(len(index)==2)
        assert(isinstance(index[0], WhichCacheTensor))
        assert(isinstance(index[1], tuple))
        assert(all(isinstance(x, WeightType) for x in index[1]))
        return True
    
    def construct_index(self, which_cache: WhichCacheTensor, weight_types: Tuple[WeightType,...]| WeightType) -> TensorIndex:
        if(isinstance(weight_types, WeightType)):
            weight_types = (weight_types,)* (self.dimension -1)
        assert(len(weight_types)== self.dimension -1)
        index = (which_cache, weight_types)
        self.validate_index(index)
        return index

    def get_metric_product(self, grid_info: GridInfo,  index: TensorIndex,  sim_params: SimParams, use_cache =True) -> npt.NDArray[np.float64]:
        self.validate_index(index)
        which_cache, weight_type_per_axis = index
        cached_product = self.cached_tensor_data.get(index, None)
        if(cached_product is not None and use_cache):
            return cached_product.array
        mesh_grid = None
        if(cached_product is not None): 
            _, mesh_grid = cached_product
        if(mesh_grid is None): 
            mesh_grid = grid_info.mesh_grid(weight_type_per_axis)[0] # NOTE: Assuming that mesh_grid is fixed for a given simulation.  Also is that the shape is the same for all axes
        expected_product_size =  self.expected_tensor_dimensions(mesh_grid, index)
        #print("DEBUG", expected_product_size, which_cache, weight_type)
        match which_cache:
            case WhichCacheTensor.METRIC:
                product = self.metric(mesh_grid,expected_product_size,sim_params)
            case WhichCacheTensor.INVERSE_METRIC:
                product = self.inv_metric(mesh_grid,expected_product_size,sim_params)
            case WhichCacheTensor.DETERMINANT:
                product = self.determinant(mesh_grid, expected_product_size,sim_params)
            case WhichCacheTensor.PARTIAL_DER:
                product = self.partial_derivative(mesh_grid, expected_product_size,sim_params)
            case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                product = self.Christoffel_upper(mesh_grid, expected_product_size,sim_params)
            case WhichCacheTensor.PARTIAL_LN_ALPHA:
                product = self.partial_ln_alpha(mesh_grid, expected_product_size,sim_params)
            case WhichCacheTensor.ALPHA:
                product = self.alpha(mesh_grid, expected_product_size,sim_params)
            case WhichCacheTensor.BETA:
                product = self.beta(mesh_grid, expected_product_size,sim_params)
            case _:
                product  = None
        if(product is None):
           raise Exception("invalid metric product", index)
        assert(product.shape== expected_product_size )
        self.cached_tensor_data[index] = CachedTensor(product, mesh_grid)
        return self.cached_tensor_data[index].array

    # Call this at the end of the constructor of the subclass to make sure that your metric conforms
    # Should also fill all of the caches (Assuming I didn't miss one)
    def sanity_check(self, grid_info: GridInfo, sim_params: SimParams) ->  bool:
        assert(self.dimension != None)
        assert(self.dimension >=2) # Need at least 1+1 formulation
        center_weights_info = tuple([WeightType.CENTER]* (self.dimension -1))
        which_tensors = [
            WhichCacheTensor.METRIC,
            WhichCacheTensor.INVERSE_METRIC,
            WhichCacheTensor.DETERMINANT,
            WhichCacheTensor.PARTIAL_DER,
            WhichCacheTensor.CHRISTOFFEL_UPPER0,
            WhichCacheTensor.PARTIAL_LN_ALPHA,
            WhichCacheTensor.ALPHA,
            WhichCacheTensor.BETA
        ]
        cases  = [(item, center_weights_info) for item in which_tensors]
        # Add edge weight determinant case
        mixed_weight_types_per_axis = []
        for i in range(self.dimension -1):
            item  = []
            for j in range(self.dimension -1):
                if(i==j):
                    item.append(WeightType.EDGE)
                else:
                    item.append(WeightType.CENTER)
            mixed_weight_types_per_axis.append( tuple(item) )
        for item in mixed_weight_types_per_axis:
            cases.append( (WhichCacheTensor.DETERMINANT, item) )
        # Fill up the caches
        for case in cases:
            print("DEBUG: Filling cache for ", case)
            self.get_metric_product(grid_info, case, sim_params, use_cache=False)
        assert(len(self.cached_tensor_data)== len(cases))
        self.get_metric_product(grid_info, case, sim_params, use_cache=True)
        assert(len(self.cached_tensor_data)== len(cases))
        for pair in self.cached_tensor_data.items():
            index, cached_tensor = pair
            axes = index[1]
            expected_mesh_grid = grid_info.mesh_grid(axes)[0]
            assert(np.array_equal(cached_tensor.mesh_grid, expected_mesh_grid))
            expected_size = self.expected_tensor_dimensions(cached_tensor.mesh_grid, index)
            assert(cached_tensor.array.shape== expected_size)
        # Lorentzian manifolds should have negative determinant everywhere, right?
        det = self.get_metric_product(grid_info,  self.construct_index(WhichCacheTensor.DETERMINANT, WeightType.CENTER),  sim_params, use_cache=True)
        assert(np.all( det<=0 ))

    def expected_tensor_dimensions(self, mesh_grid: npt.NDArray[np.float64], index: TensorIndex) -> npt.NDArray[np.int64]:
        self.validate_index(index)
        which_tensor, _ = index
        # Implicitly assuming that all mesh grids have the same shape.
        match which_tensor:
            case WhichCacheTensor.METRIC:
                # first, second, gridsize
                size = tuple([self.dimension, self.dimension, *(mesh_grid.shape)])
            case WhichCacheTensor.INVERSE_METRIC:
                # first, second, gridsize
                size  = tuple([self.dimension, self.dimension, *(mesh_grid.shape)])
            case WhichCacheTensor.DETERMINANT:
                # gridsize
                size   = tuple([*(mesh_grid.shape)])
            case WhichCacheTensor.PARTIAL_DER:
                # Derivative, first, second, gridsize
                size   = tuple([self.dimension, self.dimension, self.dimension, *(mesh_grid.shape)])
            case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                # Upper, first_lower, second_lower, gridsize
                size   = tuple([self.dimension,self.dimension, self.dimension, *(mesh_grid.shape)])
            case WhichCacheTensor.PARTIAL_LN_ALPHA:
                # derivative, gridsize
                size  = tuple([self.dimension,  *(mesh_grid.shape)])
            case WhichCacheTensor.ALPHA:
                # gridsize
                size  =tuple([*(mesh_grid.shape)])
            case WhichCacheTensor.BETA:
                # gridsize
                size   = tuple([self.dimension-1,*(mesh_grid.shape)])
            case _:
                raise Exception("Unimplemented tensor product from metric")
        return size

    def cell_weights(self, grid_info:GridInfo, weight_type: Tuple[WeightType,...]| WeightType, sim_params: SimParams): 
        determinant_index = self.construct_index(WhichCacheTensor.DETERMINANT, weight_type)
        return np.sqrt(-(self.get_metric_product( grid_info, determinant_index, sim_params, use_cache=True)))

    def weight_system(self, U_cart: npt.ArrayLike, grid_info: GridInfo, weight_type: WeightType, sim_params: SimParams):
        weights = self.cell_weights(grid_info, weight_type, sim_params)
        return weights * U_cart
    
    def unweight_system(self, U: npt.ArrayLike, grid_info: GridInfo,  weight_type: WeightType, sim_params: SimParams):
        weights = self.cell_weights(grid_info, weight_type, sim_params)
        return U/weights
    
    def three_vector_mag_squared(self, vec:npt.ArrayLike, grid_info:  GridInfo, weight_type: Tuple[WeightType,...]| WeightType, sim_params: SimParams):
        if(isinstance(weight_type, WeightType)):
            weight_type = (weight_type,)* (self.dimension -1)
        assert(len(weight_type)== self.dimension -1)
        index = (WhichCacheTensor.METRIC, weight_type )
        metric =  self.get_metric_product(grid_info, index,sim_params)
        dim_slice = [slice(1, None, None), slice(1, None, None)] # Get the spatial components of the metric tensor
        grid_slice = [slice(None)]*(self.dimension-1) # Index through all of the grid dimensions
        index = tuple(dim_slice+grid_slice)
        spatial_metric = metric[index]
        output = np.einsum("ij...,i...,j...->...", spatial_metric, vec, vec)
        return output
        # return  np.clip(output, a_min=None, a_max=1-epsilon) # Hack to prevent velocities which are way to big
    
    def W(self, alpha: npt.NDArray[np.float64], three_velocities: npt.ArrayLike, grid_info:  GridInfo, weight_type: WeightType, sim_params: SimParams):
        # NOTE: velocities Need to be the pure spatial velocities. They should **not** be the spatial components of the 4 velocity vector
        v2_mag  = self.three_vector_mag_squared(three_velocities, grid_info, weight_type, sim_params)
        np.clip(v2_mag, 0, 1.0 - 1e-14, out=v2_mag)
        inter =1-v2_mag
#        return alpha.array*np.power( np.clip(inter, a_min=0+epsilon, a_max=1), -0.5)
        return alpha*np.power( inter, -0.5)

    def three_vel_to_four_vel_components(self, three_velocities_unpadded:npt.NDArray,  grid_info: GridInfo, sim_params: SimParams)-> npt.NDArray:
        alpha_index = self.construct_index(WhichCacheTensor.ALPHA, WeightType.CENTER)
        alpha = self.get_metric_product( grid_info, alpha_index,  sim_params)
        W = self.W(alpha, three_velocities_unpadded, grid_info, WeightType.CENTER, sim_params)
        return W*(three_velocities_unpadded-self.shift_vector( sim_params, grid_info))
    
    def shift_vector(self, grid_info: GridInfo, sim_params: SimParams)-> npt.NDArray:
        beta_index = self.construct_index(WhichCacheTensor.BETA, WeightType.CENTER)
        alpha_index = self.construct_index(WhichCacheTensor.ALPHA, WeightType.CENTER)
        beta = self.get_metric_product( grid_info, beta_index,  sim_params)
        alpha = self.get_metric_product( grid_info, alpha_index,  sim_params)
        shift = beta/alpha
        return shift

    @abstractmethod
    def metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def inv_metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 

    @abstractmethod
    def determinant(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) -> npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def partial_derivative(self, mesh_grid: Tuple[npt.NDArray[np.float64],...] , expected_product_size: Tuple[int,...], sim_params: SimParams) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 


    @abstractmethod
    def Christoffel_upper(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] , sim_params: SimParams) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def partial_ln_alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] , sim_params: SimParams) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.ones(expected_product_size)
        return output 
    
    @abstractmethod
    def beta(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...], sim_params: SimParams ) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output
