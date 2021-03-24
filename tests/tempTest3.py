from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from submodlib import DisparitySumFunction
from submodlib import FeatureBasedFunction
from submodlib import GraphCutFunction
from submodlib import ClusteredFunction

num_internal_clusters = 20 #3
num_sparse_neighbors = 100 #10 #4
num_random = 15 #2
num_clusters = 20 #3
cluster_std_dev = 4 #1
num_samples = 500 #8
num_set = 20 #3
num_features = 500
metric = "euclidean"

# num_internal_clusters = 3 #3
# num_sparse_neighbors = 5 #10 #4
# num_random = 2 #2
# num_clusters = 3#3
# cluster_std_dev = 4 #1
# num_samples = 9
# num_set = 3 #3
# num_features = 2
# metric = "euclidean"



def test():

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(0,100), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))
    print(data)

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    dataArray = np.array(data)

    print("Instantiating...")
    obj1 = ClusteredFunction(n=num_samples, mode="multi", f_name="GraphCut", metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    print("Instantiated")
    
    eval1 = obj1.evaluate(set1)
    
    print("Instantiating...")
    obj2 = ClusteredFunction(n=num_samples, mode="single", f_name="GraphCut", metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    print("Instantiated")
    
    eval2 = obj2.evaluate(set1)
    
    print("Eval1: ", eval1)
    print("Eval2: ", eval2)

if __name__ == '__main__':
    test()