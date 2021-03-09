import pytest
import numpy as np
from scipy import sparse
import scipy
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel, create_cluster_kernels
from submodlib_cpp import FacilityLocation
#from submodlib_cpp import ClusteredFunction
from submodlib import ClusteredFunction

'''
data=np.array([
    [100, 21, 365, 5], 
    [57, 18, -5, -6], 
    [16, 255, 68, -8], 
    [2,20,6, 2000], 
    [12,20,68, 200]
    ])
'''
data =np.array( [(4.5,13.5), (5,13.5), (5.5,13.5), (14.5,13.5), (15,13.5), (15.5,13.5),
(4.5,13), (5,13), (5.5,13), (14.5,13), (15,13), (15.5,13),
(4.5,12.5), (5,12.5), (5.5,12.5), (14.5,12.5), (15,12.5), (15.5,12.5),
(4.5,7.5), (5,7.5), (5.5,7.5), (14.5,7.5), (15,7.5), (15.5,7.5),
(4.5,7), (5,7), (5.5,7), (14.5,7), (15,7), (15.5,7),
(4.5,6.5), (5,6.5), (5.5,6.5), (14.5,6.5), (15,6.5), (15.5,6.5),
(7.5,10), (12.5,10), (10,12.5), (10,7.5), (4.5, 15.5), (5,9.5), (5,10.5)] )

#a, b, c = create_cluster_kernels(data.tolist(), 'euclidean', num_cluster = 2)
#print(a)
#print(c)
#print(b)
#obj = ClusteredFunction(43, 'FacilityLocation', a, b, c)

obj = ClusteredFunction(n=43, f_name='FacilityLocation', data=data)

X = {1, 23, 4}
print(obj.evaluate(X))

X = {1, 23, 4}
print(obj.marginalGain(X, 0))

#print(obj.maximize('NaiveGreedy', 10, False, False, False))
print(obj.maximize(10,'NaiveGreedy', False, False, False))
