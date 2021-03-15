from sklearn.datasets import make_blobs
import random
import numpy as np
import submodlib_cpp as subcp
from submodlib.helper import create_kernel
from scipy import sparse
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib import ClusteredFunction

#prepare data to be used in the analysis
num_clusters = 10 #50
cluster_std_dev = 4
num_samples = 500 #5000
num_set = 6 #50
num_neighbors = 10 #40
num_executions=5

points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
data = list(map(tuple, points))

# get num_set data points belonging to cluster#1
random.seed(1)
cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
subset1 = random.sample(cluster1Indices, num_set)
set1 = set(subset1[:-1])

# get num_set data points belonging to different clusters
subset2 = []
for i in range(num_set):
    #find the index of first point that belongs to cluster i
    diverse_index = cluster_ids.tolist().index(i)
    subset2.append(diverse_index)
set2 = set(subset2[:-1])

item1 = subset1[-1]
item2 = subset2[-1]

dataArray = np.array(data)

##### Dense similarity kernel in CPP
def fl_dense_cpp_kernel():
    obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

########### Dense Similairty Kernel in Python
def fl_dense_py_kernel():
    _, K_dense = create_kernel(dataArray, 'dense','euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_master=False)
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

##### Sparse similarity kernel in CPP
def fl_sparse_cpp_kernel():
    obj = FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric="euclidean", num_neighbors=num_neighbors)
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

########### Sparse Similairty Kernel in Python
def fl_sparse_py_kernel():
    _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_neighbors)
    obj = FacilityLocationFunction(n=num_samples, mode="sparse", sijs=K_sparse, num_neighbors=num_neighbors)
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

##### Clustered mode BIRCH clustering
def fl_mode_birch():
    obj = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric="euclidean", num_clusters=num_clusters)
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

##### Clustered mode user clustering
def fl_mode_user():
    obj = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_labels=cluster_ids.tolist())
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

##### Clustered function BIRCH clustering multi
def fl_clustered_birch_multi():
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters)
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

##### Clustered function user clustering multi
def fl_clustered_user_multi():
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_lab=cluster_ids.tolist())
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

##### Clustered function BIRCH clustering single
def fl_clustered_birch_single():
    obj = ClusteredFunction(n=num_samples, mode="single", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters)
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

##### Clustered function user clustering single
def fl_clustered_user_single():
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_lab=cluster_ids.tolist())
    obj.maximize(num_set,'NaiveGreedy', False, False, False)

#fl_dense_cpp_kernel()
#fl_dense_py_kernel()
#fl_sparse_cpp_kernel()
#fl_sparse_py_kernel()
#fl_mode_birch()
fl_mode_user()
# fl_clustered_birch_multi()
# fl_clustered_user_multi()
# fl_clustered_birch_single()
# fl_clustered_user_single()