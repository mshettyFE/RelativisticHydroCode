from enum import Enum 
from CommonClasses import VariableSet

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

    def get_boundary_conds(self, index: int, var_set: VariableSet) -> tuple[BoundaryCondition,BoundaryCondition]:
        match var_set:
            case VariableSet.CONSERVATIVE | VariableSet.PRIMITIVE| VariableSet.VECTOR:
                assert(index>=1)
                assert(index<=len(self.left_bcs))             
                return (self.left_bcs[index-1], self.right_bcs[index-1])
            case _:
                return (BoundaryCondition.ZERO_GRAD, BoundaryCondition.ZERO_GRAD)

if __name__=="__main__":
    pass
