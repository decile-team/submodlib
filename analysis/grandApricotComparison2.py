# grandApricotComparison2.py
# Author: Vishal Kaushal
# Run as 'python grandApricotComparison2.py' to compare performance of 
# different alternatives listed in "methods" including apricot
# Uses python's timeit module

from sklearn.datasets import make_blobs
import random
import numpy as np
import submodlib.helper as helper
from submodlib.functions.facilityLocation import FacilityLocationFunction
from apricot import FacilityLocationSelection
import timeit
import csv

methods = ["fl_dense_cpp_kernel", "fl_dense_cpp_kernel_cpp", "fl_dense_py_kernel_current", "fl_dense_py_kernel_sklearn", "fl_dense_py_kernel_np_numba", "fl_dense_py_kernel_np", "fl_dense_py_kernel_fastdist", "fl_dense_py_kernel_scipy", "apricot_dense"]

##### Dense similarity kernel in CPP in Python
def fl_dense_cpp_kernel():
    obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

##### Dense similarity kernel in CPP in CPP
def fl_dense_cpp_kernel_cpp():
    obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean", create_dense_cpp_kernel_in_python=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

########### Dense Similairty Kernel in Python current
def fl_dense_py_kernel_current():
    K_dense = helper.create_kernel(dataArray, mode='dense', metric='euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

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

########### Dense Similairty Kernel in Python fastdist
def fl_dense_py_kernel_fastdist():
    K_dense = helper.create_kernel_dense_fastdist(dataArray,'euclidean')
    obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs=K_dense, separate_rep=False)
    obj.maximize(budget=budget,optimizer=optimizer, stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

########### Dense Similairty Kernel in Python scipy
def fl_dense_py_kernel_scipy():
    K_dense = helper.create_kernel_dense_scipy(dataArray,'euclidean')
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
#params = [(50, 5, 10), (100, 10, 50)]

first = True

for param in params:
    results_csv = [["Num_Samples", "Function", "Time"]]
    print("Parameters: ", param)
    #prepare data to be used in the analysis
    num_samples = param[0]
    num_clusters = param[1]
    num_neighbors = param[2]
    budget = int(num_samples/10)

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))
    dataArray = np.array(data)

    if first == True:
        print("Pre compiling np_numba, fastdist and apricot functions")
        fl_dense_py_kernel_np_numba()
        fl_dense_py_kernel_fastdist()
        apricot_dense()
        first = False
    
    for method in methods:
        print("Method: ", method)
        row=[num_samples, method]
        func = method + "()"
        setup = "from __main__ import " + method
        t = timeit.timeit(func, setup, number=num_executions)
        t = round(t/num_executions,num_places)
        row.append(t)
        results_csv.append(row)
    with open("submodlib_apricot_grand2_" + str(num_samples) + ".csv", "w") as f:
        writer = csv.writer(f)
        for result in results_csv:
            writer.writerow(result)
