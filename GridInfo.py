import numpy.typing as npt
import numpy as np
from enum import Enum

class CoordinateChoice(Enum):
    CARTESIAN = 0 
    SPHERICAL = 1

class WeightType(Enum):
    CENTER = 0 
    EDGE = 1

class GridInfo:
    leftmost_edges: npt.NDArray
    rightmost_edges: npt.NDArray
    NCells: npt.NDArray

    def __init__(self, left: npt.ArrayLike, right: npt.ArrayLike, cell_count: npt.ArrayLike):
        assert(left.shape==right.shape)
        assert(right.shape==cell_count.shape)
        assert(left.ndim==1)
        assert(right.ndim==1)
        assert(cell_count.ndim==1)
        self.leftmost_edges = left
        self.rightmost_edges = right
        self.NCells = cell_count

    def delta(self):
        return 1.0/(self.NCells+1.0) 

    def construct_grid_edges(self,index: np.uint):
        # Need +1  in order to generate NCells when using construct_grid_centers()
        return np.linspace(self.leftmost_edges[index], self.rightmost_edges[index], self.NCells[index]+1 )
    
    def construct_grid_centers(self, index: np.uint):
        grid_edges = self.construct_grid_edges(index)
        return 0.5*(grid_edges[1:]+grid_edges[:-1])

    def weights(self, coordinate_system:CoordinateChoice, weight_type: WeightType):
        # Results will get broadcast to appropriate dimension
        match coordinate_system:
            case CoordinateChoice.SPHERICAL:
                match weight_type:
                    case WeightType.CENTER:
                        r = self.construct_grid_centers(0) # Assuming the r index is in slot 0 
                    case WeightType.EDGE:
                        r = self.construct_grid_edges(0)
                    case _:
                        raise Exception("Invalid weight type")
                weights  = np.power(r, 2)
            case CoordinateChoice.CARTESIAN:
                match weight_type:
                    case WeightType.CENTER:
                        size = self.NCells[0]
                    case WeightType.EDGE:
                        size = self.NCells[0]+1
                    case _:
                        raise Exception("Invalid weight type")
                weights = np.ones(size)
            case _:
                raise Exception("Invalid coordinate_system")
        return weights
 
    def weight_system(self, U_cart: npt.ArrayLike, coordinate_system: CoordinateChoice, weight_type: WeightType):
        weights  = self.weights(coordinate_system, weight_type)
        return (weights * U_cart.T).T

    def unweight_system(self, U: npt.ArrayLike, coordinate_system: CoordinateChoice, weight_type: WeightType):
        weights  = self.weights(coordinate_system, weight_type)
        return (U.T/weights).T

if __name__=="__main__":
    pass
