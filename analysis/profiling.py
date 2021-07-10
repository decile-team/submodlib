# profiling.py
# Author: Vishal Kaushal
# Run as 'python -m cProfile -o analysis.prof profiling.py' 
# to generate profiling information for the uncommented function call
# Analyze the profile dump by running 'snakeviz analysis.prof' or 'pyprof2calltree -k -i analysis.prof'

from sklearn.datasets import make_blobs
import random
import numpy as np
import submodlib_cpp as subcp
import submodlib.helper as helper
from scipy import sparse
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib import ClusteredFunction
from apricot import FacilityLocationSelection

#prepare data to be used in the analysis
num_clusters = 10
cluster_std_dev = 2
num_samples = 10000
num_neighbors = 100
optimizer = "LazyGreedy"
num_features = 1024
budget = int(num_samples/10)

#points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
data = list(map(tuple, points))

dataArray = np.array(data)

########### Dense Similairty Kernel in Python sklearn
def fl_dense_py_kernel_sklearn():
    K_dense = helper.create_kernel_dense_sklearn(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

########### Dense Similairty Kernel in Python np_numba
def fl_dense_py_kernel_np_numba():
    K_dense = helper.create_kernel_dense_np_numba(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

########### Dense Similairty Kernel in Python np
def fl_dense_py_kernel_np():
    K_dense = helper.create_kernel_dense_np(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)


##### Dense similarity kernel in CPP
def fl_dense_cpp_kernel():
    obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

########### Dense Similairty Kernel in Python
def fl_dense_py_kernel():
    K_dense = helper.create_kernel(dataArray, mode='dense', metric='euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Sparse similarity kernel in CPP
def fl_sparse_cpp_kernel():
    obj = FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric="euclidean", num_neighbors=num_neighbors)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

########### Sparse Similairty Kernel in Python
def fl_sparse_py_kernel():
    K_sparse = helper.create_kernel(dataArray, mode='sparse', metric='euclidean', num_neigh=num_neighbors)
    obj = FacilityLocationFunction(n=num_samples, mode="sparse", sijs=K_sparse, num_neighbors=num_neighbors)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Clustered mode BIRCH clustering
def fl_mode_birch():
    obj = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric="euclidean", num_clusters=num_clusters)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Clustered mode user clustering
def fl_mode_user():
    obj = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_labels=cluster_ids.tolist())
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Clustered function BIRCH clustering multi
def fl_clustered_birch_multi():
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Clustered function user clustering multi
def fl_clustered_user_multi():
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_lab=cluster_ids.tolist())
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Clustered function BIRCH clustering single
def fl_clustered_birch_single():
    obj = ClusteredFunction(n=num_samples, mode="single", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Clustered function user clustering single
def fl_clustered_user_single():
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_lab=cluster_ids.tolist())
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def apricot_dense():
    obj = FacilityLocationSelection(n_samples=budget, metric='euclidean', optimizer='lazy')
    obj.fit_transform(dataArray)

#fl_dense_cpp_kernel()
#fl_dense_py_kernel()
#fl_sparse_cpp_kernel()
#fl_sparse_py_kernel()
#fl_mode_birch()
#fl_mode_user()
# fl_clustered_birch_multi()
# fl_clustered_user_multi()
# fl_clustered_birch_single()
# fl_clustered_user_single()
#apricot_dense()
#fl_dense_py_kernel_sklearn()
#fl_dense_py_kernel_np_numba()
fl_dense_py_kernel_np()
