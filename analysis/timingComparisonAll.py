# timingComparisonAll.py
# Author: Vishal Kaushal
# Run as 'python timingComparisonAll.py' to compare performance of different alternatives of
# invoking FacilityLocation

from sklearn.datasets import make_blobs
import random
import time
import numpy as np
import timeit
import submodlib_cpp as subcp
from submodlib.helper import create_kernel
from scipy import sparse
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib import ClusteredFunction
import pandas as pd
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

#prepare data to be used in the analysis
num_samples = 5000 #50 100 200 500 1000 5000
num_clusters = 10 #5 10 10 10 10 10
cluster_std_dev = 2
num_set = 9 #4 9 9 9 9 9  #should be <= num_clusters and <= num points in each cluster
num_neighbors = 100 #10 50 100 100 100
num_executions = 3
num_places = 6
num_features = 1024

# points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
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

results_csv = [["Function", "Create", "Eval1", "Eval2", "Gain1", "Gain2", "SetM", "EvalFast", "GainFast", "Maximize"]]

print("Timing the dense and sparse kernel creations in Python and C++...")

row = []
def py_dense_kernel():
    K_dense = create_kernel(dataArray, mode='dense', metric='euclidean')
    return K_dense
# t_obj1 = timeit.Timer('py_dense_kernel()', 'from __main__ import py_dense_kernel')
# t = t_obj1.timeit(number = 1)
# l_record.append(("py_dense_kernel:obj(str)", t))
# t_obj2 = timeit.Timer(py_dense_kernel)
# t = t_obj2.timeit(number = 1)
# l_record.append(("py_dense_kernel:obj(callable)", t))
t, pyDenseKernel = timeit.timeit('py_dense_kernel()', 'from __main__ import py_dense_kernel', number=num_executions)
row.append(("py_dense_kernel", round(t/num_executions,num_places)))
results_csv.append(row)

row=[]
def cpp_dense_kernel():
    content = np.array(subcp.create_kernel(dataArray.tolist(), "euclidean", np.shape(dataArray)[0]))
    val = content[0]
    row = list(map(lambda arg: int(arg), content[1]))
    col = list(map(lambda arg: int(arg), content[2]))
    sijs = np.zeros((num_samples,num_samples))
    sijs[row,col] = val
    return sijs
t, _ = timeit.timeit('cpp_dense_kernel()', 'from __main__ import cpp_dense_kernel', number=num_executions)
row.append(("cpp_dense_kernel", round(t/num_executions,num_places)))
results_csv.append(row)

row=[]
def py_sparse_kernel():
    K_sparse = create_kernel(dataArray, mode='sparse', metric='euclidean', num_neigh=num_neighbors)
    return K_sparse
t, pySparseKernel = timeit.timeit('py_sparse_kernel()', 'from __main__ import py_sparse_kernel', number=num_executions)
row.append(("py_sparse_kernel", round(t/num_executions,num_places)))
results_csv.append(row)

row=[]
def cpp_sparse_kernel():
    content = np.array(subcp.create_kernel(dataArray.tolist(), "euclidean", num_neighbors))
    val = content[0]
    row = list(map(lambda arg: int(arg), content[1]))
    col = list(map(lambda arg: int(arg), content[2]))
    sijs = sparse.csr_matrix((val, (row, col)), [num_samples,num_samples])
    return sijs
t, _ = timeit.timeit('cpp_sparse_kernel()', 'from __main__ import cpp_sparse_kernel', number=num_executions)
row.append(("cpp_sparse_kernel", round(t/num_executions,num_places)))
results_csv.append(row)

def maximize(obj):
    obj.maximize(budget=num_set,optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)

def evaluate(obj, subset):
    obj.evaluate(subset)

def marginalGain(obj, subset, item):
    obj.marginalGain(subset, item)

def setMemoization(obj, subset):
    obj.setMemoization(subset)

def evaluateWithMemoization(obj, subset):
    obj.evaluateWithMemoization(subset)

def marginalGainWithMemoization(obj, subset, item):
    obj.marginalGainWithMemoization(subset, item)

##### Dense similarity kernel in CPP

print("Timing FL with creating dense similarity kernel in CPP...")

row = ["fl_dense_cpp_kernel"]

def create_fl_dense_cpp_kernel():
    return FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
t, fl_dense_cpp_kernel_obj = timeit.timeit('create_fl_dense_cpp_kernel()', 'from __main__ import create_fl_dense_cpp_kernel', number=num_executions)
#l_record.append(("create_fl_dense_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_dense_cpp_kernel_obj,set1)', 'from __main__ import evaluate, fl_dense_cpp_kernel_obj, set1', number=num_executions)
#l_record.append(("eval_fl_dense_cpp_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_dense_cpp_kernel_obj,set2)', 'from __main__ import evaluate, fl_dense_cpp_kernel_obj, set2', number=num_executions)
#l_record.append(("eval_fl_dense_cpp_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_dense_cpp_kernel_obj,set1, item1)', 'from __main__ import marginalGain, fl_dense_cpp_kernel_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_dense_cpp_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_dense_cpp_kernel_obj,set1, item2)', 'from __main__ import marginalGain, fl_dense_cpp_kernel_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_dense_cpp_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_dense_cpp_kernel_obj,set1)', 'from __main__ import setMemoization, fl_dense_cpp_kernel_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_dense_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_dense_cpp_kernel_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_dense_cpp_kernel_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_dense_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_dense_cpp_kernel_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_dense_cpp_kernel_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_dense_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_dense_cpp_kernel_obj)', 'from __main__ import maximize, fl_dense_cpp_kernel_obj', number=num_executions)
#l_record.append(("max_fl_dense_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

########### Dense Similairty Kernel in Python
print("Timing FL with pre created Python dense similarity kernel...")

row = ["fl_dense_py_kernel"]

def create_fl_dense_py_kernel():
    return FacilityLocationFunction(n=num_samples, mode="dense", sijs=pyDenseKernel, separate_rep=False)
t, fl_dense_py_kernel_obj = timeit.timeit('create_fl_dense_py_kernel()', 'from __main__ import create_fl_dense_py_kernel', number=num_executions)
#l_record.append(("create_fl_dense_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_dense_py_kernel_obj,set1)', 'from __main__ import evaluate, fl_dense_py_kernel_obj, set1', number=num_executions)
#l_record.append(("eval_fl_dense_py_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_dense_py_kernel_obj,set2)', 'from __main__ import evaluate, fl_dense_py_kernel_obj, set2', number=num_executions)
#l_record.append(("eval_fl_dense_py_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_dense_py_kernel_obj,set1, item1)', 'from __main__ import marginalGain, fl_dense_py_kernel_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_dense_py_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_dense_py_kernel_obj,set1, item2)', 'from __main__ import marginalGain, fl_dense_py_kernel_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_dense_py_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_dense_py_kernel_obj,set1)', 'from __main__ import setMemoization, fl_dense_py_kernel_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_dense_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_dense_py_kernel_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_dense_py_kernel_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_dense_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_dense_py_kernel_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_dense_py_kernel_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_dense_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_dense_py_kernel_obj)', 'from __main__ import maximize, fl_dense_py_kernel_obj', number=num_executions)
#l_record.append(("max_fl_dense_py_kernel", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

##### Sparse similarity kernel in CPP
print("Timing FL with sparse similarity kernel in CPP...")

row = ["fl_sparse_cpp_kernel"]

def create_fl_sparse_cpp_kernel():
    return FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric="euclidean", num_neighbors=num_neighbors)
t, fl_sparse_cpp_kernel_obj = timeit.timeit('create_fl_sparse_cpp_kernel()', 'from __main__ import create_fl_sparse_cpp_kernel', number=num_executions)
#l_record.append(("create_fl_sparse_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_sparse_cpp_kernel_obj,set1)', 'from __main__ import evaluate, fl_sparse_cpp_kernel_obj, set1', number=num_executions)
#l_record.append(("eval_fl_sparse_cpp_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_sparse_cpp_kernel_obj,set2)', 'from __main__ import evaluate, fl_sparse_cpp_kernel_obj, set2', number=num_executions)
#l_record.append(("eval_fl_sparse_cpp_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_sparse_cpp_kernel_obj,set1, item1)', 'from __main__ import marginalGain, fl_sparse_cpp_kernel_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_sparse_cpp_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_sparse_cpp_kernel_obj,set1, item2)', 'from __main__ import marginalGain, fl_sparse_cpp_kernel_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_sparse_cpp_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_sparse_cpp_kernel_obj,set1)', 'from __main__ import setMemoization, fl_sparse_cpp_kernel_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_sparse_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_sparse_cpp_kernel_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_sparse_cpp_kernel_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_sparse_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_sparse_cpp_kernel_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_sparse_cpp_kernel_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_sparse_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_sparse_cpp_kernel_obj)', 'from __main__ import maximize, fl_sparse_cpp_kernel_obj', number=num_executions)
#l_record.append(("max_fl_sparse_cpp_kernel", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

########### Sparse Similairty Kernel in Python
print("Timing FL with pre-created Python sparse similarity kernel...")

row = ["fl_sparse_py_kernel"]

def create_fl_sparse_py_kernel():
    return FacilityLocationFunction(n=num_samples, mode="sparse", sijs=pySparseKernel, num_neighbors=num_neighbors)
t, fl_sparse_py_kernel_obj = timeit.timeit('create_fl_sparse_py_kernel()', 'from __main__ import create_fl_sparse_py_kernel', number=num_executions)
#l_record.append(("create_fl_sparse_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_sparse_py_kernel_obj,set1)', 'from __main__ import evaluate, fl_sparse_py_kernel_obj, set1', number=num_executions)
#l_record.append(("eval_fl_sparse_py_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_sparse_py_kernel_obj,set2)', 'from __main__ import evaluate, fl_sparse_py_kernel_obj, set2', number=num_executions)
#l_record.append(("eval_fl_sparse_py_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_sparse_py_kernel_obj,set1, item1)', 'from __main__ import marginalGain, fl_sparse_py_kernel_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_sparse_py_kernel1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_sparse_py_kernel_obj,set1, item2)', 'from __main__ import marginalGain, fl_sparse_py_kernel_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_sparse_py_kernel2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_sparse_py_kernel_obj,set1)', 'from __main__ import setMemoization, fl_sparse_py_kernel_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_sparse_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_sparse_py_kernel_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_sparse_py_kernel_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_sparse_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_sparse_py_kernel_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_sparse_py_kernel_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_sparse_py_kernel", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_sparse_py_kernel_obj)', 'from __main__ import maximize, fl_sparse_py_kernel_obj', number=num_executions)
#l_record.append(("max_fl_sparse_py_kernel", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

##### Clustered mode BIRCH clustering
print("Timing FL clustered mode with BIRCH clustering...")

row = ["fl_mode_birch"]

def create_fl_mode_birch():
    return FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric="euclidean", num_clusters=num_clusters)
t, fl_mode_birch_obj = timeit.timeit('create_fl_mode_birch()', 'from __main__ import create_fl_mode_birch', number=num_executions)
#l_record.append(("create_fl_mode_birch", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_mode_birch_obj,set1)', 'from __main__ import evaluate, fl_mode_birch_obj, set1', number=num_executions)
#l_record.append(("eval_fl_mode_birch1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_mode_birch_obj,set2)', 'from __main__ import evaluate, fl_mode_birch_obj, set2', number=num_executions)
#l_record.append(("eval_fl_mode_birch2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_mode_birch_obj,set1, item1)', 'from __main__ import marginalGain, fl_mode_birch_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_mode_birch1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_mode_birch_obj,set1, item2)', 'from __main__ import marginalGain, fl_mode_birch_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_mode_birch2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_mode_birch_obj,set1)', 'from __main__ import setMemoization, fl_mode_birch_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_mode_birch", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_mode_birch_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_mode_birch_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_mode_birch", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_mode_birch_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_mode_birch_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_mode_birch", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_mode_birch_obj)', 'from __main__ import maximize, fl_mode_birch_obj', number=num_executions)
#l_record.append(("max_fl_mode_birch", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

##### Clustered mode user clustering
print("Timing FL clustered mode with user clustering...")

row = ["fl_mode_user"]

def create_fl_mode_user():
    return FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_labels=cluster_ids.tolist())
t, fl_mode_user_obj = timeit.timeit('create_fl_mode_user()', 'from __main__ import create_fl_mode_user', number=num_executions)
#l_record.append(("create_fl_mode_user", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_mode_user_obj,set1)', 'from __main__ import evaluate, fl_mode_user_obj, set1', number=num_executions)
#l_record.append(("eval_fl_mode_user1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_mode_user_obj,set2)', 'from __main__ import evaluate, fl_mode_user_obj, set2', number=num_executions)
#l_record.append(("eval_fl_mode_user2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_mode_user_obj,set1, item1)', 'from __main__ import marginalGain, fl_mode_user_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_mode_user1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_mode_user_obj,set1, item2)', 'from __main__ import marginalGain, fl_mode_user_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_mode_user2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_mode_user_obj,set1)', 'from __main__ import setMemoization, fl_mode_user_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_mode_user", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_mode_user_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_mode_user_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_mode_user", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_mode_user_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_mode_user_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_mode_user", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_mode_user_obj)', 'from __main__ import maximize, fl_mode_user_obj', number=num_executions)
#l_record.append(("max_fl_mode_user", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

##### Clustered function BIRCH clustering multi
print("Timing FL clustered function with BIRCH clustering multi kernels...")

row = ["fl_clustered_birch_multi"]

def create_fl_clustered_birch_multi():
    return ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters)
t, fl_clustered_birch_multi_obj = timeit.timeit('create_fl_clustered_birch_multi()', 'from __main__ import create_fl_clustered_birch_multi', number=num_executions)
#l_record.append(("create_fl_clustered_birch_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_birch_multi_obj,set1)', 'from __main__ import evaluate, fl_clustered_birch_multi_obj, set1', number=num_executions)
#l_record.append(("eval_fl_clustered_birch_multi1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_birch_multi_obj,set2)', 'from __main__ import evaluate, fl_clustered_birch_multi_obj, set2', number=num_executions)
#l_record.append(("eval_fl_clustered_birch_multi2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_birch_multi_obj,set1, item1)', 'from __main__ import marginalGain, fl_clustered_birch_multi_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_clustered_birch_multi1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_birch_multi_obj,set1, item2)', 'from __main__ import marginalGain, fl_clustered_birch_multi_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_clustered_birch_multi2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_clustered_birch_multi_obj,set1)', 'from __main__ import setMemoization, fl_clustered_birch_multi_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_clustered_birch_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_clustered_birch_multi_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_clustered_birch_multi_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_clustered_birch_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_clustered_birch_multi_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_clustered_birch_multi_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_clustered_birch_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_clustered_birch_multi_obj)', 'from __main__ import maximize, fl_clustered_birch_multi_obj', number=num_executions)
#l_record.append(("max_fl_clustered_birch_multi", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

##### Clustered function user clustering multi
print("Timing FL clustered function with user clustering multi kernels...")

row = ["fl_clustered_user_multi"]

def create_fl_clustered_user_multi():
    return ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_lab=cluster_ids.tolist())
t, fl_clustered_user_multi_obj = timeit.timeit('create_fl_clustered_user_multi()', 'from __main__ import create_fl_clustered_user_multi', number=num_executions)
#l_record.append(("create_fl_clustered_user_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_user_multi_obj,set1)', 'from __main__ import evaluate, fl_clustered_user_multi_obj, set1', number=num_executions)
#l_record.append(("eval_fl_clustered_user_multi1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_user_multi_obj,set2)', 'from __main__ import evaluate, fl_clustered_user_multi_obj, set2', number=num_executions)
#l_record.append(("eval_fl_clustered_user_multi2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_user_multi_obj,set1, item1)', 'from __main__ import marginalGain, fl_clustered_user_multi_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_clustered_user_multi1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_user_multi_obj,set1, item2)', 'from __main__ import marginalGain, fl_clustered_user_multi_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_clustered_user_multi2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_clustered_user_multi_obj,set1)', 'from __main__ import setMemoization, fl_clustered_user_multi_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_clustered_user_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_clustered_user_multi_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_clustered_user_multi_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_clustered_user_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_clustered_user_multi_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_clustered_user_multi_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_clustered_user_multi", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_clustered_user_multi_obj)', 'from __main__ import maximize, fl_clustered_user_multi_obj', number=num_executions)
#l_record.append(("max_fl_clustered_user_multi", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

##### Clustered function BIRCH clustering single
print("Timing FL clustered function with BIRCH clustering single kernel...")

row = ["fl_clustered_birch_single"]

def create_fl_clustered_birch_single():
    return ClusteredFunction(n=num_samples, mode="single", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters)
t, fl_clustered_birch_single_obj = timeit.timeit('create_fl_clustered_birch_single()', 'from __main__ import create_fl_clustered_birch_single', number=num_executions)
#l_record.append(("create_fl_clustered_birch_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_birch_single_obj,set1)', 'from __main__ import evaluate, fl_clustered_birch_single_obj, set1', number=num_executions)
#l_record.append(("eval_fl_clustered_birch_single1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_birch_single_obj,set2)', 'from __main__ import evaluate, fl_clustered_birch_single_obj, set2', number=num_executions)
#l_record.append(("eval_fl_clustered_birch_single2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_birch_single_obj,set1, item1)', 'from __main__ import marginalGain, fl_clustered_birch_single_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_clustered_birch_single1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_birch_single_obj,set1, item2)', 'from __main__ import marginalGain, fl_clustered_birch_single_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_clustered_birch_single2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_clustered_birch_single_obj,set1)', 'from __main__ import setMemoization, fl_clustered_birch_single_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_clustered_birch_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_clustered_birch_single_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_clustered_birch_single_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_clustered_birch_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_clustered_birch_single_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_clustered_birch_single_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_clustered_birch_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_clustered_birch_single_obj)', 'from __main__ import maximize, fl_clustered_birch_single_obj', number=num_executions)
#l_record.append(("max_fl_clustered_birch_single", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

##### Clustered function user clustering single
print("Timing FL clustered function with user clustering single kernel...")

row = ["fl_clustered_user_single"]

def create_fl_clustered_user_single():
    return ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', data=dataArray, metric="euclidean", num_clusters=num_clusters, cluster_lab=cluster_ids.tolist())
t, fl_clustered_user_single_obj = timeit.timeit('create_fl_clustered_user_single()', 'from __main__ import create_fl_clustered_user_single', number=num_executions)
#l_record.append(("create_fl_clustered_user_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_user_single_obj,set1)', 'from __main__ import evaluate, fl_clustered_user_single_obj, set1', number=num_executions)
#l_record.append(("eval_fl_clustered_user_single1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluate(fl_clustered_user_single_obj,set2)', 'from __main__ import evaluate, fl_clustered_user_single_obj, set2', number=num_executions)
#l_record.append(("eval_fl_clustered_user_single2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_user_single_obj,set1, item1)', 'from __main__ import marginalGain, fl_clustered_user_single_obj, set1, item1', number=num_executions)
#l_record.append(("gain_fl_clustered_user_single1", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGain(fl_clustered_user_single_obj,set1, item2)', 'from __main__ import marginalGain, fl_clustered_user_single_obj, set1, item2', number=num_executions)
#l_record.append(("gain_fl_clustered_user_single2", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('setMemoization(fl_clustered_user_single_obj,set1)', 'from __main__ import setMemoization, fl_clustered_user_single_obj, set1', number=num_executions)
#l_record.append(("setMem_fl_clustered_user_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('evaluateWithMemoization(fl_clustered_user_single_obj,set1)', 'from __main__ import evaluateWithMemoization, fl_clustered_user_single_obj, set1', number=num_executions)
#l_record.append(("evalFast_fl_clustered_user_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('marginalGainWithMemoization(fl_clustered_user_single_obj,set1,item1)', 'from __main__ import marginalGainWithMemoization, fl_clustered_user_single_obj, set1,item1', number=num_executions)
#l_record.append(("gainFast_fl_clustered_user_single", t))
row.append(round(t/num_executions, num_places))

t, _ = timeit.timeit('maximize(fl_clustered_user_single_obj)', 'from __main__ import maximize, fl_clustered_user_single_obj', number=num_executions)
#l_record.append(("max_fl_clustered_user_single", t))
row.append(round(t/num_executions, num_places))

results_csv.append(row)

# df = pd.DataFrame(columns = ['name', 'time'],data=l_record)
# print(df)

with open("timing_results.csv", "w") as f:
    writer = csv.writer(f)
    for result in results_csv:
        writer.writerow(result)