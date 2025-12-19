import numpy.typing as npt
import numpy as np
from enum import Enum
from typing import Tuple, List

class WeightType(Enum):
    CENTER = 0 
    EDGE = 1

class Scaling(Enum):
    LINEAR = 0
    LOG10 = 1

class GridInfo:
    leftmost_edges: npt.NDArray
    rightmost_edges: npt.NDArray
    NCells: npt.NDArray
    scaling: list
    deltas: List[npt.NDArray]

    def __init__(self, left: npt.ArrayLike, right: npt.ArrayLike, cell_count: npt.ArrayLike, scalings: list):
        assert(left.shape==right.shape)
        assert(right.shape==cell_count.shape)
        assert(left.ndim==1)
        assert(right.ndim==1)
        assert(cell_count.ndim==1)
        assert(len(scalings)==left.shape[0])
        self.leftmost_edges = left
        self.rightmost_edges = right
        self.NCells = cell_count
        self.scaling = scalings
        self.deltas = self.construct_deltas()

    def n_cells_by_weight_type(self, Weight_type: WeightType, index: np.uint) -> np.uint:
        match Weight_type:
            case WeightType.CENTER:
                return self.NCells[index]
            case WeightType.EDGE:
                return self.NCells[index]+1
            case _:
                raise Exception("Invalid Weight type")

    def delta(self, index: np.uint):
        return self.deltas[index]

    def construct_deltas(self):
        output = []
        for index, _ in enumerate(self.NCells):
            edges = self.construct_grid_edges(index)
            right = edges[1:]
            left = edges[:-1]
            output.append(right-left)
        return output
    
    def construct_grid_edges(self,index: np.uint):
        n_cells = self.n_cells_by_weight_type(WeightType.EDGE, index)
        match self.scaling[index]:
            case Scaling.LINEAR:
                return np.linspace(self.leftmost_edges[index], self.rightmost_edges[index], n_cells)
            case Scaling.LOG10:
                return np.logspace(self.leftmost_edges[index], self.rightmost_edges[index], n_cells )
            case _:
                raise Exception("Unimplemented scaling")
    
    def construct_grid_centers(self, index: np.uint):
        grid_edges = self.construct_grid_edges(index)
        return 0.5*(grid_edges[1:]+grid_edges[:-1])

    def mesh_grid(self, weights_per_axis: Tuple[WeightType,...]):
        assert(len(weights_per_axis)==len(self.NCells))
        assert(all(isinstance(x, WeightType) for x in weights_per_axis))
        boundaries = []
        for i, weight_type in enumerate(weights_per_axis):
            match weight_type:
                case WeightType.CENTER:
                    boundaries.append( self.construct_grid_centers(i) )
                case WeightType.EDGE:
                    boundaries.append( self.construct_grid_edges(i) )
                case _:
                    raise Exception("Invalid Weight type")
        out = np.meshgrid(*boundaries,indexing='ij')
        return out

if __name__=="__main__":
    pass
