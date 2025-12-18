from enum import Enum 

class BoundaryCondition(Enum):
    FIXED = 0 
    ZERO_GRAD = 1
    REFLECTIVE = 2

class BoundaryConditionManager:
    left_bcs: list[BoundaryCondition]
    right_bcs: list[BoundaryCondition]

    def __init__(self, left_boundaries: list[BoundaryCondition], right_boundaries: list[BoundaryCondition]):
        assert(len(left_boundaries) == len(right_boundaries))
        for left, right in zip(left_boundaries, right_boundaries):
            assert(isinstance(left, BoundaryCondition))
            assert(isinstance(right, BoundaryCondition))
        self.left_bcs = left_boundaries 
        self.right_bcs = right_boundaries 

    def get_boundary_conds(self, index: int) -> tuple[BoundaryCondition,BoundaryCondition]:
        if ( (index>=0) and (index < len(self.left_bcs))):
            return (self.left_bcs[index], self.right_bcs[index])
        else:
            return (BoundaryCondition.ZERO_GRAD, BoundaryCondition.ZERO_GRAD) # Hack to deal with arrays like the metric tensor

if __name__=="__main__":
    pass
