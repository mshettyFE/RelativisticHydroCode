from enum import Enum
from dataclasses import dataclass
import numpy as np

class TimeUpdateType(Enum):
    EULER = 0 
    RK3 = 1

class SpatialUpdateType(Enum):
    FLAT = 0 
    PLM = 1

@dataclass
class SpatialUpdate:
    method: SpatialUpdateType 
    params: dict[str, np.float64]


if __name__=="__main__":
    pass
