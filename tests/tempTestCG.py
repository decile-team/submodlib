from sklearn.datasets import make_blobs
import numpy as np
import random
from submodlib import FacilityLocationConditionalGainFunction
from submodlib.helper import create_kernel
import matplotlib.pyplot as plt

num_internal_clusters = 3
num_clusters = 3 #20
cluster_std_dev = 1 #4
num_samples = 12 #500
num_features = 2 #500
metric = "euclidean"
budget = 4 #20
num_queries = 2
magnificationLambda = 2
privacyHardness=2
num_set = 3

points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)

print("points: ", points)
print("cluster_ids: ", cluster_ids)
    
pointsMinusQuery = list(map(tuple, points)) 

queries = []
query_features = []
random_clusters = random.sample(range(num_clusters), num_queries)
for c in range(num_queries): #select 10 query points
    crand = random_clusters[c]
    q_ind = cluster_ids.tolist().index(crand) #find the ind of first point that belongs to cluster crand
    queries.append(q_ind)
    query_features.append(tuple(points[q_ind]))
    pointsMinusQuery.remove(tuple(points[q_ind]))

print("queries=", queries)
print("query_features=", query_features)
print("pointsMinusQuery=", pointsMinusQuery)

# get a subset with num_set data points
random.seed(1)
set1 = random.sample(range(num_samples-num_queries), num_set)

print("set1=", set1)

imageData = np.array(pointsMinusQuery)
queryData = np.array(query_features)

num_data = num_samples-num_queries

print("n=", num_data)
print("num_privates=", num_queries)
print("image_data=", imageData)
print("privateData=", queryData)
print("set1=", set1)

obj1 = LogDeterminantConditionalGainFunction(n=num_data, num_privates=num_queries, imageData=imageData, privateData=queryData, metric=metric, privacyHardness=privacyHardness)

print("Instantiated")

subset = set()
for elem in set1:
    obj1.updateMemoization(subset, elem)
    subset.add(elem)
print("subset=", subset)
simpleEval = obj1.evaluate(subset)
fastEval = obj1.evaluateWithMemoization(subset)
print("simpleEval=", simpleEval)
print("fastEval=", fastEval)



