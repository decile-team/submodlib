#Currently it contains test for only evaluate() and marginalGain() of dense because C++ implementaion is currently
#for dense mode only.
import pytest
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import scipy
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel
import math


@pytest.fixture
def f_1():# A simple easy to calculate test case
    M = np.array([
        [1,3,2],
        [5,4,3],
        [4,7,5]
        ]) 
    obj = FacilityLocationFunction(n=3, sijs=M)
    return obj

@pytest.fixture
def f_2(): #Boundary case of just one element
    M = np.array([-0.78569]) 
    obj = FacilityLocationFunction(n=1, sijs=M)
    return obj

@pytest.fixture
def f_3(): #more realistic test case 
    M = np.array([ 
        [-0.78569, 0.75, 0.9, -0.56, 0.005], 
        [0.00006, 0.400906, -0.203, 0.9765, -0.9], 
        [0.1, 0.3, 0.5, 0.0023, 0.9], 
        [-0.1, 0.1, 0.1405, 0.0023, 0.3], 
        [-0.123456, 0.0789, 0.00456, 0.001, -0.9]
        ]) 
    obj = FacilityLocationFunction(n=5, sijs=M)
    return obj



'''This is the similarity matrix corrosponding to data matrix created in f_5() below
[[ 1.        ,  0.18480505,  0.31649794,  0.01689587,  0.34060725],
[ 0.18480505,  1.        ,  0.32673257, -0.09584317, -0.03871451],
[ 0.31649794,  0.32673257,  1.        , -0.01977083,  0.14792658],
[ 0.01689587, -0.09584317, -0.01977083,  1.        ,  0.94295957],
[ 0.34060725, -0.03871451,  0.14792658,  0.94295957,  1.        ],]
'''
@pytest.fixture
def f_5():
    data=np.array([
    [100, 21, 365, 5], 
    [57, 18, -5, -6], 
    [16, 255, 68, -8], 
    [2,20,6, 2000], 
    [12,20,68, 200]
    ])
    obj = FacilityLocationFunction(n=5, data=data, mode="dense", metric="cosine")
    return obj


class TestFL:
    
    #Testing wrt similarity matrix
    def test_1_1(self, f_1):
        X = {1}
        assert f_1.evaluate(X)==14

    def test_1_2(self, f_1):
        X = {0,2}
        assert f_1.evaluate(X)==12

    def test_1_3(self, f_1):
        X = {0,2}
        item =1
        assert f_1.marginalGain(X, item)==3

    def test_2_1(self, f_2):
        X = {0}
        assert math.isclose(round(f_2.evaluate(X),5), -0.78569)

    
    def test_2_2(self, f_2):
        X = {0}
        item = 0
        assert f_2.marginalGain(X, item)==0

    def test_3_1(self, f_3):
        X = {0,2,4}
        assert math.isclose(round(f_3.evaluate(X),5), 2.10462)
    
    def test_3_2(self, f_3):
        X = {1,3}
        assert math.isclose(round(f_3.evaluate(X),4), 2.2054)

    def test_3_3(self, f_3):
        X = {0,2,4}
        item =1
        assert math.isclose(round(f_3.marginalGain(X, item),6), 0.475186)


    #Testing wrt data
    def test_5_1(self, f_5): 
        X = {1}
        assert math.isclose(round(f_5.evaluate(X),2), 1.38)
    
    def test_5_2(self, f_5):
        X = {0,2}
        assert math.isclose(round(f_5.evaluate(X),2), 2.68)

    def test_5_3(self, f_5):
        X = {0,2}
        item = 1
        assert math.isclose(round(f_5.marginalGain(X, item),3), 0.673)

    #Negative Test cases:
    def test_4_1(self): #Non-square dense similarity matrix 
        M = np.array([[1,2,3], [4,5,6]])
        FacilityLocationFunction(n=2, sijs=M)
    
    def test_4_2(self): #Inconsistency between n and no of examples in M
        M = np.array([[1,2,3], [4,5,6]])
        FacilityLocationFunction(n=1, sijs=M)

    def test_4_3(self): # X not a subset of ground set for evaluate()
        M = np.array([[1,2], [3,4]])
        obj = FacilityLocationFunction(n=2, sijs=M)
        X={0,2}
        obj.evaluate(X)

    def test_4_4(self): # X not a subset of ground set for marginalGain()
        M = np.array([[1,2], [3,4]])
        obj = FacilityLocationFunction(n=2, sijs=M)
        X={0,2}
        obj.marginalGain(X, 1)

    def test_4_5(self): # If sparse matrix is provided but without providing number of neighbors that were used to create it
        data = np.array([[1,2], [3,4]])
        num_neigh, M = create_kernel(data, 'sparse','euclidean', num_neigh=1)
        FacilityLocationFunction(n=2, sijs=M) #its important for user to pass num_neigh with sparse matrix because otherwise
                                              #there is no way for Python FL and C++ FL to know how many nearest neighours were
                                              #reatined in sparse matrix
        
    def test_4_6(self): # n==0
        data = np.array([[1,2], [3,4]])
        num_neigh, M = create_kernel(data, 'sparse','euclidean', num_neigh=1)
        FacilityLocationFunction(n=0, sijs=M, num_neigh=num_neigh)
    



    



    
    





        