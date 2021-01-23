#A dryrun of implemented code with dummy data
import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel


data = np.array([[1,2, 3], [3,4, 5], [4, 5,6]])


#dryrun of create_kernel
n_, K_dense = create_kernel(data, 'dense','euclidean')
print(K_dense)
n_, K_sparse = create_kernel(data, 'sparse','euclidean', num_neigh=2)
print(K_sparse)


#dryrun of C++ FL and Python FL when user provides similarity matrix
#1) with dense matrix
obj = FacilityLocationFunction(n=3, sijs = K_dense, seperateMaster=False)
X = {1}
print(obj.evaluate(X))
X = {1,2}
print(obj.evaluate(X))
X = {1}
print(obj.marginalGain(X,2))

#2) with sparse matrix 
obj = FacilityLocationFunction(n=3, sijs = K_sparse, num_neigh=2)


#dryrun of C++ FL and Python FL when user provides data
#1) with dense mode
obj = FacilityLocationFunction(n=3, data=data, mode="dense", metric="euclidean")
X = {1}
print(obj.evaluate(X))
X = {1,2}
print(obj.evaluate(X))
X = {1}
print(obj.marginalGain(X,2))

#2) with sparse mode
obj = FacilityLocationFunction(n=3, data=data, mode="sparse", metric="euclidean")
