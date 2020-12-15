#Timing Analysis for C++ and Python helper code
from timeit import timeit 
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import scipy
import submodlib_cpp as subcp
from submodlib.helper import create_kernel
import random


def f(arg):
    random.seed(arg)
    x=random.randrange(1000)-random.randrange(1000)
    y=random.randrange(10)-random.randrange(10)
    z=(random.randrange(100)/100)-(random.randrange(100)/100)
    return (x,y,z)

l = list(map(f, range(100000)))


#data = np.array([[0, 1, 3, 5], [5, 1, 5, -6], [10, 2, 6, -8], [12,20,68, 200], [12,20,68, 200]])
#data = np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]])
data = np.array(l)
num_neigh=2
#metric = "euclidean"

def fun1():# cpp_helper_euclidean (Non-vectorized, min-heap based approach)
    subcp.create_kernel(data.tolist(), "euclidean" ,num_neigh)

def fun2(): #python_helper_euclidean(vectorized knn clustering approach) 
    ED = euclidean_distances(data) 
    gamma = 1/np.shape(data)[1] 
    ES = np.exp(-ED* gamma) 
    nbrs = NearestNeighbors(n_neighbors=num_neigh, metric="euclidean").fit(data)
    _, ind = nbrs.kneighbors(data)
    ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
    row, col = zip(*ind_l)
    mat = np.zeros(np.shape(ES))
    mat[row, col]=1
    ES_ = ES*mat


def fun3():# cpp_helper_cosine (Non-vectorized, min-heap based approach)
    subcp.create_kernel(data.tolist(), "cosine" ,num_neigh)


def fun4(): #python_helper_cosine(vectorized knn clustering approach) 
    CS = cosine_similarity(data) 
    nbrs = NearestNeighbors(n_neighbors=num_neigh, metric="cosine").fit(data)
    _, ind = nbrs.kneighbors(data)
    ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
    row, col = zip(*ind_l)
    mat = np.zeros(np.shape(CS))
    mat[row, col]=1
    CS_ = CS*mat

print("cpp_helper_euclidean:", timeit('fun1', 'from __main__ import fun1'),'\n')
print("python_helper_euclidean:", timeit('fun2', 'from __main__ import fun2'),'\n')
print("cpp_helper_cosine:", timeit('fun3', 'from __main__ import fun3'),'\n')
print("python_helper_cosine:", timeit('fun4', 'from __main__ import fun4'),'\n')