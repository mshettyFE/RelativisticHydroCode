import numpy.typing as npt
import numpy as np
from enum import Enum

class WeightType(Enum):
    CENTER = 0 
    EDGE = 1

class GridInfo:
    leftmost_edges: npt.NDArray
    rightmost_edges: npt.NDArray
    NCells: npt.NDArray
    meshgrid_center: npt.NDArray
    meshgrid_edge:  npt.NDArray

    def __init__(self, left: npt.ArrayLike, right: npt.ArrayLike, cell_count: npt.ArrayLike):
        assert(left.shape==right.shape)
        assert(right.shape==cell_count.shape)
        assert(left.ndim==1)
        assert(right.ndim==1)
        assert(cell_count.ndim==1)
        self.leftmost_edges = left
        self.rightmost_edges = right
        self.NCells = cell_count
        self.meshgrid_center = self.mesh_grid(WeightType.CENTER) 
        self.meshgrid_edge  = self.mesh_grid(WeightType.EDGE)

    def delta(self):
        return 1.0/(self.NCells+1.0) 

    def construct_grid_edges(self,index: np.uint):
        # Need +1  in order to generate NCells when using construct_grid_centers()
        return np.linspace(self.leftmost_edges[index], self.rightmost_edges[index], self.NCells[index]+1 )
    
    def construct_grid_centers(self, index: np.uint):
        grid_edges = self.construct_grid_edges(index)
        return 0.5*(grid_edges[1:]+grid_edges[:-1])

    def mesh_grid(self, weight_type: WeightType):
        boundaries = []
        match weight_type:
            case WeightType.CENTER:
                boundaries = [self.construct_grid_centers(i) for i in range(len(self.NCells))]
            case WeightType.EDGE:
                boundaries  = [self.construct_grid_edges(i) for i in range(len(self.NCells))]
            case _:
                raise Exception("Invalid Weight type")
        out = np.meshgrid(*boundaries,indexing='ij')
        return out

if __name__=="__main__":
    pass
