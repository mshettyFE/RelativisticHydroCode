import numpy as np 
from abc import ABC,abstractmethod 
from dataclasses import dataclass 
import numpy.typing as npt
from GridInfo import WeightType, GridInfo
from enum import Enum
from typing import Tuple, List

class WhichCacheTensor(Enum):
    METRIC =  0
    INVERSE_METRIC  =  1
    DETERMINANT =  2
    PARTIAL_DER = 3
    CHRISTOFFEL_UPPER0  = 4
    PARTIAL_LN_ALPHA = 5
    ALPHA  = 6

class METRIC_VARIABLE_INDEX(Enum):
    TIME = 0
    SPACE_1 = 1
    SPACE_2 = 2
    SPACE_3 = 3

@dataclass
class cached_array:
    array: npt.NDArray[np.float64]   = None 
    weight_type: WeightType = WeightType.CENTER

class Metric(ABC):
    dimension: np.float64  = None 
    cached_metric_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_metric_edge: cached_array  = cached_array(None, WeightType.EDGE)
    cached_inv_metric_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_inv_metric_edge: cached_array  = cached_array(None, WeightType.EDGE)
    cached_determinant_center:  cached_array  = cached_array(None, WeightType.CENTER)
    cached_determinant_edge: cached_array = cached_array(None, WeightType.EDGE)
    cached_partial_der_metric_center: cached_array  = cached_array(None, WeightType.CENTER)
    cached_christoffel_upper0_center: cached_array  = cached_array(None, WeightType.CENTER)
    cached_partial_ln_alpha_center: cached_array =  cached_array(None, WeightType.CENTER)
    cached_alpha_center: cached_array =  cached_array(None, WeightType.CENTER)

    # Call this at the end of the constructor of the subclass to make sure that your metric conforms
    # Should also fill all of the caches (Assuming I didn't miss one)
    def sanity_check(self, grid_info: GridInfo) ->  bool:
        assert(self.dimension != None)
        assert(self.dimension >=2) # Need at least 1+1 formulation
        cases = [
            (WhichCacheTensor.METRIC, WeightType.CENTER),
            (WhichCacheTensor.METRIC, WeightType.EDGE),
            (WhichCacheTensor.INVERSE_METRIC, WeightType.CENTER),
            (WhichCacheTensor.INVERSE_METRIC, WeightType.EDGE),
            (WhichCacheTensor.DETERMINANT, WeightType.CENTER),
            (WhichCacheTensor.DETERMINANT, WeightType.EDGE),
            (WhichCacheTensor.PARTIAL_DER, WeightType.CENTER),
            (WhichCacheTensor.CHRISTOFFEL_UPPER0, WeightType.CENTER)  ,          
            (WhichCacheTensor.PARTIAL_LN_ALPHA, WeightType.CENTER)  ,          
            (WhichCacheTensor.ALPHA, WeightType.CENTER)           
        ]

        # Fill up the caches
        for item in cases:
            self.get_metric_product(grid_info, item[0], item[1], use_cache=False)
        # Lorentzian manifolds should have negative determinant everywhere, right?
        det = self.get_metric_product(grid_info, WhichCacheTensor.DETERMINANT, WeightType.CENTER, use_cache=True).array
        assert(np.all( det<=0 ))

    def verify_meshgrid(self, mesh_grid:Tuple[npt.NDArray[np.float64],...]) ->  bool:
        assert(len(mesh_grid) == self.dimension-1) # Ignore time axis. Only dealing with time independent metrics
        assert(all(x.shape == mesh_grid[0].shape for x in mesh_grid))
        return True
    
    def retrieve_cache(self, weight_type: WeightType, which_cache: WhichCacheTensor)-> Tuple[cached_array, bool] | Tuple[str, bool]:
        output = None
        match weight_type:
            case WeightType.CENTER:
                match which_cache:
                    case WhichCacheTensor.METRIC:
                        output = (self.cached_metric_center, True)
                    case WhichCacheTensor.INVERSE_METRIC:
                        output = (self.cached_inv_metric_center, True)
                    case WhichCacheTensor.DETERMINANT:
                        output = (self.cached_determinant_center, True)
                    case WhichCacheTensor.PARTIAL_DER:
                        output = (self.cached_partial_der_metric_center, True)
                    case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                        output = (self.cached_christoffel_upper0_center, True)
                    case WhichCacheTensor.PARTIAL_LN_ALPHA:
                        output = (self.cached_partial_ln_alpha_center, True)
                    case WhichCacheTensor.ALPHA:
                        output = (self.cached_alpha_center, True)
                    case _:
                        output =  ("Unimplemented tensor product from metric", False)
            case WeightType.EDGE:
                match which_cache:
                    case WhichCacheTensor.METRIC:
                        output = (self.cached_metric_edge, True)
                    case WhichCacheTensor.INVERSE_METRIC:
                        output = (self.cached_inv_metric_edge, True)
                    case WhichCacheTensor.DETERMINANT:
                        output = (self.cached_determinant_edge, True)
                    case WhichCacheTensor.PARTIAL_DER:
                        output =  ("You shouldn't need to calculate metric partial derivatives on cell boundaries", False)
                    case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                        output =  ("You shouldn't need to calculate Chistoffel symbols on cell boundaries", False)
                    case WhichCacheTensor.PARTIAL_LN_ALPHA:
                        output =  ("You shouldn't need to calculate partial ln alpha on cell boundaries", False)
                    case WhichCacheTensor.ALPHA:
                        output =  ("You shouldn't need to calculate alpha on cell boundaries", False)
                    case _:
                        output =  ("Unimplemented tensor product from metric", False)
            case _:
                output = ("Invalid weight type", False)
        # Output can possibly be (None, True). Need to handle that case
        return output

    def expected_tensor_dimensions(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], which_cache: WhichCacheTensor) -> Tuple[List[int], bool] | Tuple[str, bool]:
        match which_cache:
            case WhichCacheTensor.METRIC:
                # first, second, gridsize
                return (tuple([self.dimension, self.dimension, *(mesh_grid[0].shape)]), True)
            case WhichCacheTensor.INVERSE_METRIC:
                # first, second, gridsize
                return (tuple([self.dimension, self.dimension, *(mesh_grid[0].shape)]), True)
            case WhichCacheTensor.DETERMINANT:
                # gridsize
                return (tuple([*(mesh_grid[0].shape)]), True)
            case WhichCacheTensor.PARTIAL_DER:
                # Derivative, first, second, gridsize
                return (tuple([self.dimension, self.dimension, self.dimension, *(mesh_grid[0].shape)]), True)
            case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                # Upper, first_lower, second_lower, gridsize
                return (tuple([self.dimension,self.dimension, self.dimension, *(mesh_grid[0].shape)]), True)
            case WhichCacheTensor.PARTIAL_LN_ALPHA:
                # derivative, gridsize
                return (tuple([self.dimension,  *(mesh_grid[0].shape)]), True)
            case WhichCacheTensor.ALPHA:
                # gridsize
                return (tuple([*(mesh_grid[0].shape)]), True)
            case _:
                return ("Unimplemented tensor product from metric", False)

    def store_cache(self, new_tensor: npt.NDArray[np.float64],  weight_type: WeightType, which_cache: WhichCacheTensor) -> Tuple[cached_array, bool] | Tuple[str, bool]:
        # Make sure that new_tensor has the same shape as the target cache. Can use self.expected_tensor_dimension() to check for this
        match weight_type:
            case WeightType.CENTER:
                match which_cache:
                    case WhichCacheTensor.METRIC:
                        self.cached_metric_center.array = new_tensor
                        return (self.cached_metric_center, True)
                    case WhichCacheTensor.INVERSE_METRIC:
                        self.cached_inv_metric_center.array = new_tensor
                        return (self.cached_inv_metric_center,True)
                    case WhichCacheTensor.DETERMINANT:
                        self.cached_determinant_center.array  = new_tensor
                        return (self.cached_determinant_center,True)
                    case WhichCacheTensor.PARTIAL_DER:
                        self.cached_partial_der_metric_center.array  = new_tensor
                        return (self.cached_partial_der_metric_center,True)
                    case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                        self.cached_christoffel_upper0_center.array  = new_tensor
                        return (self.cached_christoffel_upper0_center,True)
                    case WhichCacheTensor.PARTIAL_LN_ALPHA:
                        self.cached_partial_ln_alpha_center.array = new_tensor
                        return (self.cached_partial_ln_alpha_center,True)
                    case WhichCacheTensor.ALPHA:
                        self.cached_alpha_center.array = new_tensor
                        return (self.cached_alpha_center,True)
                    case _:
                        return ("Unimplemented tensor product from metric", False)
            case WeightType.EDGE:
                match which_cache:
                    case WhichCacheTensor.METRIC:
                        self.cached_metric_edge.array = new_tensor
                        return(self.cached_metric_edge, True)
                    case WhichCacheTensor.INVERSE_METRIC:
                        self.cached_inv_metric_edge.array = new_tensor
                        return (self.cached_inv_metric_edge, True)
                    case WhichCacheTensor.DETERMINANT:
                        self.cached_determinant_edge.array = new_tensor
                        return (self.cached_determinant_edge, True)
                    case WhichCacheTensor.PARTIAL_DER:
                        return  ("YOu shouldn't need to calculate metric partial derivatives on cell boundaries", False)
                    case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                        return ("YOu shouldn't need to calculate Chistoffel symbols on cell boundaries", False)
                    case WhichCacheTensor.PARTIAL_LN_ALPHA:
                        return ("YOu shouldn't need to calculate partial ln alpha on cell boundaries", False)
                    case WhichCacheTensor.ALPHA:
                        return ("YOu shouldn't need to calculate alpha on cell boundaries", False)
                    case _:
                        return ("Unimplemented tensor product from metric", False)
            case _:
                return ("Invalid weight type", False)

    def cell_weights(self, grid_info:GridInfo, weight_type: WeightType):
        return np.sqrt(-(self.get_metric_product( grid_info, WhichCacheTensor.DETERMINANT,weight_type=weight_type, use_cache=True).array))

    def weight_system(self, U_cart: npt.ArrayLike, grid_info: GridInfo, weight_type: WeightType):
        weights = self.cell_weights(grid_info, weight_type)
        return (weights.T * U_cart.T).T
    
    def unweight_system(self, U: npt.ArrayLike, grid_info: GridInfo,  weight_type: WeightType):
        weights = self.cell_weights(grid_info, weight_type)
        return (U.T/weights.T).T
    
    def boost_field(self, W:npt.ArrayLike, grid_info:  GridInfo, weight_type: WeightType,alpha=1):
        # W is a ([ncells]*ndim,prim) where the prim denotes (\rho, P, and  up to 3 velocities)
        metric =  self.get_metric_product(grid_info, WhichCacheTensor.METRIC,  weight_type).array
        alpha =  self.get_metric_product(grid_info, WhichCacheTensor.ALPHA,  weight_type).array
        dim_slice = [slice(1, None, None), slice(1, None, None)] # Get the spatial components of the metric tensor
        grid_slice = [slice(None)]*(self.dimension-1) # Index through all of the grid dimensions
        index = tuple(dim_slice+grid_slice)
        spatial_metric = np.einsum("ij...->...ij",metric[index]) # Size of (dim,dim, gridsize)
        velocities = W[...,2:] # Assuming velocities are the trailing variables. Size of (gridsize, dim)
        right = np.matvec(spatial_metric, velocities) # Sum over last index. Size is (gridsize, dim)
        mag = np.vecdot(velocities, right)
        return alpha*np.power(1-mag, -0.5)
    
    def get_metric_product(self, grid_info: GridInfo, which_cache: WhichCacheTensor,  weight_type: WeightType, use_cache =True) -> cached_array:
        if(use_cache):
            output, success = self.retrieve_cache(weight_type, which_cache)
            if (success==False):
                raise Exception(output)
            maybe_output: cached_array = output
            if(maybe_output is not None):
                return output
        # Output was None, or we aren't using the cache. Need to generate the requested product
        mesh_grid = grid_info.mesh_grid(weight_type)
        expected_product_size, success =  self.expected_tensor_dimensions(mesh_grid, which_cache)
        print("DEBUG", expected_product_size, which_cache, weight_type)
        if(success==False): 
            raise Exception(expected_product_size) # Throw error message
        product = None
        match which_cache:
            case WhichCacheTensor.METRIC:
                product = self.metric(mesh_grid,expected_product_size)
            case WhichCacheTensor.INVERSE_METRIC:
                product = self.inv_metric(mesh_grid,expected_product_size)
            case WhichCacheTensor.DETERMINANT:
                product = self.determinant(mesh_grid, expected_product_size)
            case WhichCacheTensor.PARTIAL_DER:
                product = self.partial_derivative(mesh_grid, expected_product_size)
            case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                product = self.Christoffel_upper(mesh_grid, expected_product_size)
            case WhichCacheTensor.PARTIAL_LN_ALPHA:
                product = self.partial_ln_alpha(mesh_grid, expected_product_size)
            case WhichCacheTensor.ALPHA:
                product = self.alpha(mesh_grid, expected_product_size)
            case _:
                product  = None
        if(product is None):
           raise Exception("invalid metric product", which_cache,  weight_type)
        assert(product.shape== expected_product_size )
        final, success = self.store_cache(product, weight_type, which_cache)
        if(success==False):
            raise Exception(final)
        return final

    @abstractmethod
    def metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...]) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def inv_metric(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...]) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 

    @abstractmethod
    def determinant(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] ) -> npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def partial_derivative(self, mesh_grid: Tuple[npt.NDArray[np.float64],...] , expected_product_size: Tuple[int,...]) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 


    @abstractmethod
    def Christoffel_upper(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] ) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def partial_ln_alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] ) ->  npt.NDArray[np.float64]:
        # Fix upper index to 0 since that's the only one that's relevant for this problem
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def alpha(self, mesh_grid: Tuple[npt.NDArray[np.float64],...], expected_product_size: Tuple[int,...] ) ->  npt.NDArray[np.float64]:
        # Fix upper index to 0 since that's the only one that's relevant for this problem
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.ones(expected_product_size)
        return output 