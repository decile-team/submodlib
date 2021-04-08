from sklearn.datasets import make_blobs
import numpy as np
import random
from submodlib import FacilityLocationMutualInformationFunction
from submodlib.helper import create_kernel
import matplotlib.pyplot as plt

num_samples = 500
num_clusters = 10
num_features = 2
cluster_std_dev = 1
metric = "euclidean"
num_queries = 3
budget = 3

points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)

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

print("Data: ")
print(type(pointsMinusQuery))
print(pointsMinusQuery)

print("Queries:")
print(queries)
print(type(query_features))
print(query_features)

groundxs = [x[0] for x in pointsMinusQuery]
groundys = [x[1] for x in pointsMinusQuery]

queryxs = [x[0] for x in query_features]
queryys = [x[1] for x in query_features]


obj1 = FacilityLocationMutualInformationFunction(n=(num_samples-num_queries), num_queries=num_queries, imageData=np.array(pointsMinusQuery), queryData=np.array(query_features), metric=metric)

print("Instantiated")

greedyList = obj1.maximize(budget=budget,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=True)
greedyXs = [groundxs[x[0]] for x in greedyList]
greedyYs = [groundys[x[0]] for x in greedyList]

plt.scatter(groundxs, groundys, s=50, facecolors='none', edgecolors='black', label="Images")
plt.scatter(queryxs, queryys, s=50, color='green', label="Queries")
plt.scatter(greedyXs, greedyYs, s=50, color='blue', label="Greedy Set")

plt.show()

# _, imageKernel = create_kernel(pointsMinusQuery, mode="dense", metric=metric)
# queryKernel = create_kernel(query_features, mode="dense", metric=metric, X_master=pointsMinusQuery)

# obj2 = FacilityLocationMutualInformationFunction(n=num_samples, num_queries=num_queries, image_sijs=imageKernel, query_sijs=queryKernel)



