from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random
import time
import numpy as np
from submodlib import DisparitySumFunction
from submodlib import FeatureBasedFunction
from submodlib import GraphCutFunction
from submodlib import ClusteredFunction

# num_internal_clusters = 20 #3
# num_sparse_neighbors = 100 #10 #4
# num_random = 15 #2
# num_clusters = 20 #3
# cluster_std_dev = 4 #1
# num_samples = 500 #8
# num_set = 20 #3
# num_features = 500
# metric = "euclidean"
# num_sparse_neighbors_full = num_samples #because less than this doesn't work for DisparitySum


num_internal_clusters = 3 #3
num_sparse_neighbors = 5 #10 #4
num_random = 2 #2
num_clusters = 3#3
cluster_std_dev = 4 #1
num_samples = 9
num_set = 3 #3
num_features = 2
metric = "euclidean"
num_sparse_neighbors_full = num_samples #because less than this doesn't work for DisparitySum



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

    # obj = DisparitySumFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)

    # obj = FeatureBasedFunction(n=num_samples, features=data, numFeatures=num_features, sparse=False)

    #obj = GraphCutFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)
    print("Instantiating...")
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name="DisparitySum", metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    print("Instantiated")
    groundSet = obj.getEffectiveGroundSet()
    eval = obj.evaluate(groundSet)
    print(f"Eval on groundset = {eval}")
    elems = random.sample(set1, num_random)
    subset = set(elems[:-1])
    elem = elems[-1]
    print(f"Setting memoization for {subset}")
    obj.setMemoization(subset)
    zeroEval = obj.evaluate(subset)
    print(f"evaluate(subset) = {zeroEval}")
    firstEval = obj.evaluateWithMemoization(subset)
    print(f"evaluateWithMemoization(subset) = {firstEval}")
    print(f"Element to be added is {elem}")
    subset.add(elem)
    secondEval = obj.evaluate(subset)
    print(f"evaluate(subset++) = {secondEval}")
    naiveGain = secondEval - firstEval
    print("naiveGain:", naiveGain)
    subset.remove(elem)
    simpleGain = obj.marginalGain(subset, elem)
    print(f"gain(subset,elem) = {simpleGain}")
    fastGain = obj.marginalGainWithMemoization(subset, elem)
    print(f"gainFast(subset,elem) = {fastGain}")

if __name__ == '__main__':
    test()