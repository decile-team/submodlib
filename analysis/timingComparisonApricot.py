from sklearn.datasets import make_blobs
import random
from apricot import FacilityLocationSelection
import time
import numpy as np
import timeit
import submodlib_cpp as subcp
from submodlib.helper import create_kernel
from scipy import sparse
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib import ClusteredFunction
import csv

#override timeit's template to support for returning value from timed function
timeit.template = """
def inner(_it, _timer{init}):
    {setup}
    _t0 = _timer()
    for _i in _it:
        retval = {stmt}
    _t1 = _timer()
    return _t1 - _t0, retval
"""

def py_dense_kernel(dataArray):
    #print("Calling py dense kernel with ", dataArray.shape[0], " elements in ground set")
    K_dense = create_kernel(dataArray, mode='dense', metric='euclidean')
    return K_dense

def cpp_dense_kernel(dataArray):
    content = np.array(subcp.create_kernel(dataArray.tolist(), "euclidean", np.shape(dataArray)[0]))
    val = content[0]
    row = list(map(lambda arg: int(arg), content[1]))
    col = list(map(lambda arg: int(arg), content[2]))
    sijs = np.zeros((num_samples,num_samples))
    sijs[row,col] = val
    return sijs

def py_sparse_kernel(dataArray, num_neighbors):
    K_sparse = create_kernel(dataArray, mode='sparse', metric='euclidean', num_neigh=num_neighbors)
    return K_sparse

def cpp_sparse_kernel(dataArray, num_neighbors):
    content = np.array(subcp.create_kernel(dataArray.tolist(), "euclidean", num_neighbors))
    val = content[0]
    row = list(map(lambda arg: int(arg), content[1]))
    col = list(map(lambda arg: int(arg), content[2]))
    sijs = sparse.csr_matrix((val, (row, col)), [num_samples,num_samples])
    return sijs

def maximize(obj, budget):
    #print("Maximize called with budget: ", budget)
    obj.maximize(budget=budget,optimizer='LazyGreedy')

def create_fl_dense_cpp_kernel(num_samples, dataArray):
    return FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")

def create_fl_dense_py_kernel(num_samples, pyDenseKernel):
    return FacilityLocationFunction(n=num_samples, mode="dense", sijs=pyDenseKernel, separate_rep=False)

def create_fl_sparse_cpp_kernel(num_samples, dataArray, num_neighbors):
    return FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric="euclidean", num_neighbors=num_neighbors)

def create_fl_sparse_py_kernel(num_samples, pySparseKernel, num_neighbors):
    return FacilityLocationFunction(n=num_samples, mode="sparse", sijs=pySparseKernel, num_neighbors=num_neighbors)

def create_apricot_dense(budget):
    return FacilityLocationSelection(n_samples=budget, metric='euclidean', optimizer='lazy')

def apricot_fit_transform(obj, dataArray):
    obj.fit_transform(dataArray)

def create_apricot_sparse(budget, num_neighbors):
    return FacilityLocationSelection(n_samples=budget, metric='euclidean', optimizer='lazy', n_neighbors=num_neighbors)

cluster_std_dev = 2
num_executions = 5
num_places = 6
num_features = 1024

params = [(50, 5, 10), (100, 10, 50), (200, 10, 100), (500, 10, 100), (1000, 10, 100), (5000, 10, 100)]
results_csv = [["Num_Samples", "Function", "Create", "Maximize"]]

for param in params:
    print("Parameters: ", param)
    #prepare data to be used in the analysis
    num_samples = param[0]
    num_clusters = param[1]
    num_neighbors = param[2]
    budget = int(num_samples/10)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))
    dataArray = np.array(data)
    
    print("Timing the dense and sparse kernel creations in Python and C++...")
    
    row=[num_samples, "py_dense_kernel"]
    t, pyDenseKernel = timeit.timeit('py_dense_kernel(dataArray)', 'from __main__ import py_dense_kernel, dataArray', number=num_executions)
    time_pyDenseKernel = round(t/num_executions,num_places)
    row.append(time_pyDenseKernel)
    results_csv.append(row)

    row=[num_samples, "cpp_dense_kernel"]
    t, _ = timeit.timeit('cpp_dense_kernel(dataArray)', 'from __main__ import cpp_dense_kernel, dataArray', number=num_executions)
    row.append(round(t/num_executions,num_places))
    results_csv.append(row)

    row=[num_samples, "py_sparse_kernel"]
    t, pySparseKernel = timeit.timeit('py_sparse_kernel(dataArray, num_neighbors)', 'from __main__ import py_sparse_kernel, dataArray, num_neighbors', number=num_executions)
    time_pySparseKernel = round(t/num_executions,num_places)
    row.append(time_pySparseKernel)
    results_csv.append(row)

    row=[num_samples, "cpp_sparse_kernel"]
    t, _ = timeit.timeit('cpp_sparse_kernel(dataArray, num_neighbors)', 'from __main__ import cpp_sparse_kernel, dataArray, num_neighbors', number=num_executions)
    row.append(round(t/num_executions,num_places))
    results_csv.append(row)

    ##### Dense similarity kernel in CPP

    print("Timing FL with creating dense similarity kernel in CPP...")

    row = [num_samples, "fl_dense_cpp_kernel"]

    t, fl_dense_cpp_kernel_obj = timeit.timeit('create_fl_dense_cpp_kernel(num_samples, dataArray)', 'from __main__ import create_fl_dense_cpp_kernel, num_samples, dataArray', number=num_executions)
    row.append(round(t/num_executions, num_places))

    t, _ = timeit.timeit('maximize(fl_dense_cpp_kernel_obj, budget)', 'from __main__ import maximize, fl_dense_cpp_kernel_obj, budget', number=num_executions)
    row.append(round(t/num_executions, num_places))

    results_csv.append(row)

    ########### Dense Similairty Kernel in Python
    print("Timing FL with pre created Python dense similarity kernel...")

    row = [num_samples, "fl_dense_py_kernel"]

    t, fl_dense_py_kernel_obj = timeit.timeit('create_fl_dense_py_kernel(num_samples, pyDenseKernel)', 'from __main__ import create_fl_dense_py_kernel, num_samples, pyDenseKernel', number=num_executions)
    time_fl_dense_py_kernel = round(t/num_executions, num_places)
    row.append(time_fl_dense_py_kernel + time_pyDenseKernel)

    t, _ = timeit.timeit('maximize(fl_dense_py_kernel_obj, budget)', 'from __main__ import maximize, fl_dense_py_kernel_obj, budget', number=num_executions)
    row.append(round(t/num_executions, num_places))

    results_csv.append(row)

    ##### Sparse similarity kernel in CPP
    print("Timing FL with sparse similarity kernel in CPP...")

    row = [num_samples, "fl_sparse_cpp_kernel"]

    t, fl_sparse_cpp_kernel_obj = timeit.timeit('create_fl_sparse_cpp_kernel(num_samples, dataArray, num_neighbors)', 'from __main__ import create_fl_sparse_cpp_kernel, num_samples, dataArray, num_neighbors', number=num_executions)
    row.append(round(t/num_executions, num_places))

    t, _ = timeit.timeit('maximize(fl_sparse_cpp_kernel_obj, budget)', 'from __main__ import maximize, fl_sparse_cpp_kernel_obj, budget', number=num_executions)
    row.append(round(t/num_executions, num_places))

    results_csv.append(row)

    ########### Sparse Similairty Kernel in Python
    print("Timing FL with pre-created Python sparse similarity kernel...")

    row = [num_samples, "fl_sparse_py_kernel"]

    t, fl_sparse_py_kernel_obj = timeit.timeit('create_fl_sparse_py_kernel(num_samples, pySparseKernel, num_neighbors)', 'from __main__ import create_fl_sparse_py_kernel, num_samples, pySparseKernel, num_neighbors', number=num_executions)
    time_fl_sparse_py_kernel = round(t/num_executions, num_places)
    row.append(time_fl_sparse_py_kernel + time_pySparseKernel)

    t, _ = timeit.timeit('maximize(fl_sparse_py_kernel_obj, budget)', 'from __main__ import maximize, fl_sparse_py_kernel_obj, budget', number=num_executions)
    row.append(round(t/num_executions, num_places))

    results_csv.append(row)

    ##### Apricot

    print("Timing apricot dense...")

    row = [num_samples, "apricot_dense"]

    t, apricot_dense_obj = timeit.timeit('create_apricot_dense(budget)', 'from __main__ import create_apricot_dense, budget', number=num_executions)
    row.append(round(t/num_executions, num_places))

    t, _ = timeit.timeit('apricot_fit_transform(apricot_dense_obj, dataArray)', 'from __main__ import apricot_fit_transform, apricot_dense_obj, dataArray', number=num_executions)
    row.append(round(t/num_executions, num_places))

    results_csv.append(row)

    # print("Timing apricot sparse...")

    # row = [num_samples, "apricot_sparse"]

    # t, apricot_sparse_obj = timeit.timeit('create_apricot_sparse(budget, num_neighbors)', 'from __main__ import create_apricot_sparse, budget, num_neighbors', number=num_executions)
    # row.append(round(t/num_executions, num_places))

    # t, _ = timeit.timeit('apricot_fit_transform(apricot_sparse_obj, data)', 'from __main__ import apricot_fit_transform, apricot_sparse_obj, data', number=num_executions)
    # row.append(round(t/num_executions, num_places))

    # results_csv.append(row)

with open("submodlib_apricot.csv", "w") as f:
    writer = csv.writer(f)
    for result in results_csv:
        writer.writerow(result)