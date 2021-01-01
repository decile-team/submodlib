import pytest
import numpy as np
from scipy import sparse
import scipy
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel
from submodlib_cpp import FacilityLocation

data=np.array([
    [100, 21, 365, 5], 
    [57, 18, -5, -6], 
    [16, 255, 68, -8], 
    [2,20,6, 2000], 
    [12,20,68, 200]
    ])


s = {1}
obj = FacilityLocationFunction(n=5, data=data, mode="sparse", metric="cosine")

print(obj.maximize(3,'NaiveGreedy', False, False, False))
