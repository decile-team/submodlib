import pytest
import math
from sklearn.datasets import make_blobs
import numpy as np
import random
from submodlib.helper import create_kernel
from submodlib import ClusteredFunction
from submodlib import LogDeterminantFunction

# num_internal_clusters = 3 #3
# num_sparse_neighbors = 5 #10 #4
# num_random = 2 #2
# num_clusters = 3#3
# cluster_std_dev = 4 #1
# num_samples = 9
# num_set = 3 #3
# num_features = 2
# metric = "euclidean"
# #num_sparse_neighbors_full = num_samples #because less than this doesn't work for DisparitySum
# num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
# budget = 5
# num_concepts = 3

# num_internal_clusters = 5 #3
# num_sparse_neighbors = 5 #10 #4
# num_random = 2 #2
# num_clusters = 5#3
# cluster_std_dev = 4 #1
# num_samples = 15
# num_set = 3 #3
# num_features = 2
# metric = "euclidean"
# #num_sparse_neighbors_full = num_samples #because less than this doesn't work for DisparitySum
# num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
# budget = 5
# num_concepts = 3

num_internal_clusters = 20 #3
num_sparse_neighbors = 100 #10 #4
num_random = 15 #2
num_clusters = 20 #3
cluster_std_dev = 4 #1
num_samples = 500 #8
num_set = 20 #3
num_features = 500
metric = "euclidean"
#num_sparse_neighbors_full = num_samples #because less than this doesn't work for DisparitySum
num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
budget = 20
num_concepts = 50

points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
data = list(map(tuple, points))

# get num_set data points belonging to cluster#1
# random.seed(1)
# cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
# subset1 = random.sample(cluster1Indices, num_set)
# set1 = set(subset1[:-1])

# # get num_set data points belonging to different clusters
# subset2 = []
# for i in range(num_set):
#     #find the index of first point that belongs to cluster i
#     diverse_index = cluster_ids.tolist().index(i)
#     subset2.append(diverse_index)
# set2 = set(subset2[:-1])

dataArray = np.array(data)
print("Before instantiation")
# obj = ClusteredFunction(n=num_samples, mode="single", f_name="LogDeterminant", metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
obj = LogDeterminantFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full, lambdaVal=1)
print("After instantiation")
groundSet = obj.getEffectiveGroundSet()
print(groundSet)
eval = obj.evaluate(groundSet)
print(eval)