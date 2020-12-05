#Contains test for both modes (dense and sparse) and both metrics (euclidean and cosine)
import pytest
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import scipy
from submodlib.helper import create_kernel

list_tests=[
    np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]]),
    np.array([[1, 1, 1]]),
    np.array([ [1], [1], [1] ]),
    np.zeros((2,2)),
    np.ones((3,2)),
    np.identity(5),
]

class TestHelper:

    @pytest.mark.parametrize("data", list_tests)
    def test_dense_euclidean(self, data):
        ED = euclidean_distances(data) 
        gamma = 1/np.shape(data)[1] 
        ES = np.exp(-ED* gamma) #sklearn ground truth 

        assert np.array_equal(create_kernel(data, 'dense','euclidean'), ES)

    @pytest.mark.parametrize("data", list_tests)
    def test_dense_cosine(self, data):
        CS = cosine_similarity(data)  #sklearn ground truth 

        assert np.array_equal(create_kernel(data, 'dense','cosine'), CS)


    @pytest.mark.parametrize("data", list_tests)
    def test_sparse_euclidean(self, data):
        ED = euclidean_distances(data) 
        gamma = 1/np.shape(data)[1] 
        ES = np.exp(-ED* gamma)  
        ES_csr = sparse.csr_matrix(ES) #sklearn ground truth
        _, val = create_kernel(data, 'sparse','euclidean')
        assert np.array_equal(val.todense(), ES_csr.todense())

    @pytest.mark.parametrize("data", list_tests)
    def test_sparse_cosine(self, data):
        CS = cosine_similarity(data) 
        CS_csr = sparse.csr_matrix(CS) #sklearn ground truth
        _, val = create_kernel(data, 'sparse','cosine')
        assert np.array_equal(val.todense(), CS_csr.todense())


    #Negative test cases
    def test_neg1(self): # Number of neighbors more than no of data points
        data = np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]])
        create_kernel(data, 'sparse','cosine', num_neigh=6)

    def test_neg2(self): # Incorrect mode
        data = np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]])
        create_kernel(data, 'sparss','cosine', num_neigh=3)

    def test_neg3(self): # Incorrect metric
        data = np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]])
        create_kernel(data, 'sparse','cosinee', num_neigh=3)
        