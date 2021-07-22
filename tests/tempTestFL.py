from sklearn.datasets import make_blobs
import numpy as np
import random
from submodlib import ConcaveOverModularFunction
from submodlib import LogDeterminantMutualInformationFunction
from submodlib_cpp import ConcaveOverModular
from submodlib.helper import create_kernel


num_internal_clusters = 3 #3
num_sparse_neighbors = 5 #10 #4
num_random = 1 #2
num_clusters = 3#3
cluster_std_dev = 4 #1
num_samples = 10
num_set = 1 #3
num_features = 2
num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
budget = 4
num_concepts = 3
num_queries = 2
magnificationEta = 2
privacyHardness = 2
num_privates = 1
queryDiversityEta = 2
logDetLambdaVal = 1
metric = "dot"
metric_disp_sparse = "euclidean"

def data_queries():
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(50,200), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    
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
    
    # get a subset with num_set data points
    set1 = set(random.sample(range(num_samples-num_queries), num_set))

    imageData = np.array(pointsMinusQuery)
    queryData = np.array(query_features)

    return (num_samples-num_queries, num_queries, imageData, queryData, set1)


num_data, num_q, imageData, queryData, _ = data_queries()
print("Image data: ", imageData)
print("Query data: ", queryData)
imageKernel = create_kernel(imageData, mode="dense", metric=metric)
queryKernel = create_kernel(queryData, mode="dense", metric=metric, X_rep=imageData)
queryQueryKernel = create_kernel(queryData, mode="dense", metric=metric)
print("Image-Image Kernel: ", imageKernel)
print("Image-Query Kernel: ", queryKernel)
print("Query-Query Kernel: ", queryQueryKernel)
obj = LogDeterminantMutualInformationFunction(n=num_data, num_queries=num_q, data_sijs=imageKernel, query_sijs=queryKernel, query_query_sijs=queryQueryKernel, lambdaVal=logDetLambdaVal, magnificationEta=magnificationEta)
# obj = ConcaveOverModularFunction(n=num_data, num_queries=num_q, query_sijs=queryKernel, queryDiversityEta=queryDiversityEta, mode=ConcaveOverModular.squareRoot)
print("About to do naive")
input("continue?")
greedyListNaive = obj.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=True)
print("Naive done, about to do lazy")
input("conitnue?")
greedyListLazy = obj.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=True)
print("Lazy done")
naiveGains = [x[1] for x in greedyListNaive]
lazyGains = [x[1] for x in greedyListLazy]
assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"