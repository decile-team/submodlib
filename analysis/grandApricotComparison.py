# grandApricotComparison.py
# Author: Vishal Kaushal
# Run as 'python grandApricotComparison.py' to compare performance of 
# fl_dense_cpp, fl_dense_python and apricot_dense
# Uses naive time.time, runs a method num_executions times

from sklearn.datasets import make_blobs
import random
import numpy as np
import submodlib_cpp as subcp
from submodlib.helper import create_kernel
from scipy import sparse
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib import ClusteredFunction
from apricot import FacilityLocationSelection
import time
import csv

##### Dense similarity kernel in CPP
def fl_dense_cpp_kernel():
    obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

########### Dense Similairty Kernel in Python
def fl_dense_py_kernel():
    K_dense = create_kernel(dataArray, mode='dense', metric='euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def apricot_dense():
    obj = FacilityLocationSelection(n_samples=budget, metric='euclidean', optimizer='lazy')
    obj.fit_transform(dataArray)

cluster_std_dev = 2
num_executions = 3
num_places = 6
num_features = 1024
optimizer = 'LazyGreedy'

params = [(50, 5, 10), (100, 10, 50), (200, 10, 100), (500, 10, 100), (1000, 10, 100), (5000, 10, 100), (6000, 10, 100), (7000, 10, 100), (8000, 10, 100), (9000, 10, 100), (10000, 10, 100)]
results_csv = [["Num_Samples", "Function", "Time"]]

for param in params:
    print("Parameters: ", param)
    #prepare data to be used in the analysis
    num_samples = param[0]
    num_clusters = param[1]
    num_neighbors = param[2]
    budget = int(num_samples/10)

    #points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))
    dataArray = np.array(data)

    row=[num_samples, "fl_dense_cpp_kernel"]
    totalTime = 0
    for i in range(num_executions):
        start = time.time()
        fl_dense_cpp_kernel()
        stop = time.time()
        currentTime = stop - start
        totalTime += currentTime
        row.append(round(currentTime,num_places))
    actualTime = totalTime/num_executions
    row.append(round(actualTime, num_places))
    results_csv.append(row)

    row=[num_samples, "fl_dense_py_kernel"]
    totalTime = 0
    for i in range(num_executions):
        start = time.time()
        fl_dense_py_kernel()
        stop = time.time()
        currentTime = stop - start
        totalTime += currentTime
        row.append(round(currentTime,num_places))
    actualTime = totalTime/num_executions
    row.append(round(actualTime, num_places))
    results_csv.append(row)

    row=[num_samples, "apricot_dense"]
    totalTime = 0
    for i in range(num_executions):
        start = time.time()
        apricot_dense()
        stop = time.time()
        currentTime = stop - start
        totalTime += currentTime
        row.append(round(currentTime,num_places))
    actualTime = totalTime/num_executions
    row.append(round(actualTime, num_places))
    results_csv.append(row)

with open("submodlib_apricot_grand.csv", "w") as f:
    writer = csv.writer(f)
    for result in results_csv:
        writer.writerow(result)






    
        
