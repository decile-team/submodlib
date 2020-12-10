import pytest
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import scipy
import submodlib_cpp as subcp
from submodlib.helper import create_kernel


list_tests=[
    np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]]),
    np.array([[1, 1, 1]]),
    np.array([ [1], [1], [1] ]),
    np.zeros((2,2)),
    np.ones((3,2)),
    np.identity(5),
]


list_tests2=[
    np.array([[0, 1, 3, 5], [5, 1, 5, -6], [10, 2, 6, -8], [12,20,68, 200], [12,20,68, 200]]),
    np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]]),
    np.array([ [1], [1], [1] ]),
    np.zeros((2,2)),
    np.ones((3,2)),
]

class TestHelper:
    
    @pytest.mark.parametrize("data", list_tests)
    def test_euclidean_full(self, data):
        ED = euclidean_distances(data) 
        gamma = 1/np.shape(data)[1] 
        ES = np.exp(-ED* gamma) #sklearn ground truth 
        num_neigh=np.shape(data)[0]
        assert np.allclose(subcp.create_kernel(data.tolist(),'euclidean',num_neigh), ES)
    
    @pytest.mark.parametrize("data", list_tests)
    def test_cosine_full(self, data):
        CS = cosine_similarity(data)  #sklearn ground truth 
        num_neigh=np.shape(data)[0]
        assert np.allclose(subcp.create_kernel(data.tolist(),'cosine',num_neigh), CS)
        


    @pytest.mark.parametrize("data", list_tests2)
    def test_euclidean_neigh(self, data):
        ED = euclidean_distances(data) 
        gamma = 1/np.shape(data)[1] 
        ES = np.exp(-ED* gamma) 
        num_neigh=np.shape(data)[0]-1

        nbrs = NearestNeighbors(n_neighbors=num_neigh, metric="euclidean", n_jobs=3).fit(data)
        _, ind = nbrs.kneighbors(data)
        ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
        row, col = zip(*ind_l)
        mat = np.zeros(np.shape(ES))
        mat[row, col]=1
        ES_ = ES*mat #sklearn ground truth 
        
        assert np.allclose(subcp.create_kernel(data.tolist(),'euclidean',num_neigh), ES_)
    
    @pytest.mark.parametrize("data", list_tests2)
    def test_cosine_neigh(self, data):
        CS = cosine_similarity(data)  
        num_neigh=np.shape(data)[0]-1

        nbrs = NearestNeighbors(n_neighbors=num_neigh, metric="cosine", n_jobs=3).fit(data)
        _, ind = nbrs.kneighbors(data)
        ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
        row, col = zip(*ind_l)
        mat = np.zeros(np.shape(CS))
        mat[row, col]=1
        CS_ = CS*mat #sklearn ground truth 


        assert np.allclose(subcp.create_kernel(data.tolist(),'cosine',num_neigh), CS_)
      