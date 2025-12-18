from enum import Enum
import numpy as np

class TimeUpdateType(Enum):
    EULER = 0 
    RK3 = 1

class SpatialUpdateType(Enum):
    FLAT = 0 
    PLM = 1

class SpatialUpdate:
    method: SpatialUpdateType 
    params: dict[str, np.float64]

    def __init__(self, a_method, a_params):
        self.method = a_method
        self.params = a_params

    def pad_width(self) -> int:
        pad_width = None 
        match self.method:
            case SpatialUpdateType.FLAT:
                pad_width = 1
            case SpatialUpdateType.PLM:
                raise Exception("PLM spatial update not yet implemented")
            case _:
                raise Exception("Invalid spatial integration type. Can't assign pad width")
        return pad_width
 
if __name__=="__main__":
    pass
