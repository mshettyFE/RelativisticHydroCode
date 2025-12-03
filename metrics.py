import numpy as np 
from abc import ABC,abstractmethod 
from dataclasses import dataclass 
import numpy.typing as npt
from GridInfo import WeightType
from enum import Enum

class WhichCacheTensor(Enum):
    METRIC =  0
    INVERSE_METRIC  =  1
    DETERMINANT =  2
    PARTIAL_DER = 3
    CHRISTOFFEL_UPPER0  = 4

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

    # Call this at the end of the constructor of the subclass to make sure that your metric conforms
    # Should also fill all of the caches ???
    def sanity_check(self) ->  bool:
        assert(self.dimension != None)
        assert(self.dimension >=2) # Need at least 1+1 formulation
        metric = self.metric()
        expected_size = np.zeros((self.dimension, self.dimension))

    def verify_meshgrid(self, mesh_grid:tuple(np.ndarray[tuple(int), np.float64],...)) ->  bool:
        assert(len(mesh_grid) == self.dimension-1)
        assert(all(x.shape == mesh_grid[0].shape for x in mesh_grid))
        return True
    
    def retrieve_cache(self, weight_type: WeightType, which_cache: WhichCacheTensor)-> tuple(cached_array, bool) | tuple(str, bool):
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
                        output =  ("YOu shouldn't need to calculate Chistoffel symbols on cell boundaries", False)
                    case _:
                        output =  ("Unimplemented tensor product from metric", False)
            case _:
                output = ("Invalid weight type", False)
        # Output can possibly be (None, True). Need to handle that case
        return output

    def expected_tensor_dimensions(self, mesh_grid: tuple(np.ndarray[tuple(int), np.float64],...), which_cache: WhichCacheTensor) -> tuple(list(int), bool) | tuple(str, bool):
        match which_cache:
            case WhichCacheTensor.METRIC:
                return ([self.dimension, self.dimension, *(mesh_grid[0].shape)], True)
            case WhichCacheTensor.INVERSE_METRIC:
                return ([self.dimension, self.dimension, *(mesh_grid[0].shape)], True)
            case WhichCacheTensor.DETERMINANT:
                return ([*(mesh_grid[0].shape)], True)
            case WhichCacheTensor.PARTIAL_DER:
                return ([self.dimension, self.dimension, self.dimension, *(mesh_grid[0].shape)], True)
            case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                return ([self.dimension, self.dimension, *(mesh_grid[0].shape)], True)
            case _:
                return ("Unimplemented tensor product from metric", False)

    def store_cache(self, new_tensor: npt.NDArray[np.float64],  weight_type: WeightType, which_cache: WhichCacheTensor) -> tuple(cached_array, bool) | tuple(str, bool):
        # Make sure that new_tensor has the same shape as the target cache. Can use self.expected_tensor_dimension() to check for this
        match weight_type:
            case WeightType.CENTER:
                match which_cache:
                    case WhichCacheTensor.METRIC:
                        self.cached_metric_center = new_tensor
                        return (self.cached_metric_center, True)
                    case WhichCacheTensor.INVERSE_METRIC:
                        self.cached_inv_metric_center.array = new_tensor
                        return (new_tensor,True)
                    case WhichCacheTensor.DETERMINANT:
                        self.cached_determinant_center.array  = new_tensor
                        return (new_tensor,True)
                    case WhichCacheTensor.PARTIAL_DER:
                        self.cached_partial_der_metric_center.array  = new_tensor
                        return (new_tensor,True)
                    case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                        self.cached_christoffel_upper0_center.array  = new_tensor
                        return (new_tensor,True)
                    case _:
                        return ("Unimplemented tensor product from metric", False)
            case WeightType.EDGE:
                match which_cache:
                    case WhichCacheTensor.METRIC:
                        self.cached_metric_edge = new_tensor
                        return(new_tensor, True)
                    case WhichCacheTensor.INVERSE_METRIC:
                        self.cached_inv_metric_edge = new_tensor
                        return (new_tensor, True)
                    case WhichCacheTensor.DETERMINANT:
                        self.cached_determinant_edge = new_tensor
                        return (new_tensor, True)
                    case WhichCacheTensor.PARTIAL_DER:
                        return  ("YOu shouldn't need to calculate metric partial derivatives on cell boundaries", False)
                    case WhichCacheTensor.CHRISTOFFEL_UPPER0:
                        return ("YOu shouldn't need to calculate Chistoffel symbols on cell boundaries", False)
                    case _:
                        return ("Unimplemented tensor product from metric", False)
            case _:
                return ("Invalid weight type", False)

    def get_metric_product(self, which_cache: WhichCacheTensor, mesh_grid: tuple(np.ndarray[tuple(int), np.float64],...), weight_type: WeightType, use_cache =True) -> cached_array:
        if(use_cache):
            output, success = self.retrieve_cache(weight_type, which_cache)
            if (success==False):
                raise Exception(output)
            maybe_output: cached_array = output
            if(maybe_output.array != None):
                return output
        # Output was None, or we aren't using the cache. Need to generate the requested product
        expected_product_size, success =  self.expected_tensor_dimensions(mesh_grid, which_cache)
        if(success==False): 
            raise Exception(expected_product_size) # Throw error message
        product = None
        match which_cache:
            case WhichCacheTensor.METRIC:
                product = self.metric(mesh_grid,expected_product_size)
            case WhichCacheTensor.INVERSE_METRIC:
                product = self.inv_metric(mesh_grid,expected_product_size)
            case WhichCacheTensor.DETERMINANT:
                product - self.determinant(mesh_grid, expected_product_size)
            case WhichCacheTensor.PARTIAL_DER:
                product = self.partial_derivative(mesh_grid, expected_product_size)
            case WhichCacheTensor.PARTIAL_DER:
                product = self.partial_derivative(mesh_grid, expected_product_size)
            case _:
                product  = None
        if(product == None):
           raise Exception("invalid metric product")
        assert(product.shape== expected_product_size )
        final, success = self.store_cache(product, weight_type, which_cache)
        if(success==False):
            raise Exception(final)
        return final

    @abstractmethod
    def metric(self, mesh_grid: tuple(np.ndarray[tuple(int), np.float64],...), expected_product_size: tuple(int,...)) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def inv_metric(self, mesh_grid: tuple(np.ndarray[tuple(int), np.float64],...), expected_product_size: tuple(int,...)) ->  npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 

    @abstractmethod
    def determinant(self, mesh_grid: tuple(np.ndarray[tuple(int), np.float64],...), expected_product_size: tuple(int,...) ) -> npt.NDArray[np.float64]:
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
    
    @abstractmethod
    def partial_derivative(self, mesh_grid: tuple(np.ndarray[tuple(int), np.float64],...) , expected_product_size: tuple(int,...)) ->  npt.NDArray[np.float64]:
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 


    @abstractmethod
    def Christoffel_upper0(self, mesh_grid: tuple(np.ndarray[tuple(int), np.float64],...), expected_product_size: tuple(int,...) ) ->  npt.NDArray[np.float64]:
        # Fix upper index to 0 since that's the only one that's relevant for this problem
        ## NOTE : Will only deal with time independent metrics. Hence time derivatives are automatically 0
        # Use self.expected_tensor_dimension() to generate expected_product_size
        output = np.zeros(expected_product_size)
        return output 
 
    # def weight_system(self, U_cart: npt.ArrayLike, weight_type: WeightType):
    #     weights  = self.weights(weight_type)
    #     return (weights * U_cart.T).T
    #
    # def unweight_system(self, U: npt.ArrayLike, weight_type: WeightType):
    #     weights  = self.weights(weight_type)
    #     return (U.T/weights).T

