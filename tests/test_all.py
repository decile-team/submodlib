import pytest
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import scipy
from submodlib_cpp import FacilityLocation, ClusteredFunction, DisparitySum 
#from submodlib import ClusteredFunction, FacilityLocationFunction
from submodlib.helper import create_kernel
import math
import random
import copy
from sklearn.datasets import make_blobs
random.seed(1)

l_name = ["FacilityLocation", "ClusteredFunction", "DisparitySum"]

#@pytest.fixture
#def f_test_content():
num_clusters = 10
cluster_std_dev = 4
points, cluster_ids = make_blobs(n_samples=100, centers=num_clusters, n_features=10, cluster_std=cluster_std_dev, center_box=(0,100))
data = list(map(tuple, points))
dataArray = np.array(data)
ground_sub = {-1} #Some dummy filler value for C++ argument

l_ind = [el for el in range(0,100)]
random.shuffle(l_ind)
l_order1 = l_ind[0:10].copy()
l_order2 = l_order1.copy()
random.shuffle(l_order2)
l_order = [l_order1, l_order2]

#l_K = [
 #   create_kernel(dataArray, 'dense','euclidean'),
  #  ]
_, K_dense = create_kernel(dataArray, 'dense','euclidean')

l_fun=[
    FacilityLocation(100, "dense", K_dense.tolist(), 100, False, ground_sub, False),
    DisparitySum(100, "dense", K_dense.tolist(), 100, False, ground_sub)
    ]


class TestAll:
    @pytest.mark.parametrize("obj", l_fun)
    def test_order(self, obj): #Testing that order of insertions doesn't affect memoization
        flag = [True, True]

        for ord_id in range(2):
            set_ = set()
            ev_prev = obj.evaluate(set_)
            gainFast_prev = obj.marginalGainSequential(set_, l_order[ord_id][0])
            gain_prev = obj.marginalGain(set_, l_order[ord_id][0])
            obj.sequentialUpdate(set_, l_order[ord_id][0])
            set_.add(l_order[ord_id][0])
            #print(ev_prev, gainFast_prev)
            #for order in l_order:
            count=0
            for i in range(1, len(l_order[ord_id])):
                ev_curr = obj.evaluate(set_)
                gainFast_curr = obj.marginalGainSequential(set_, l_order[ord_id][i])    
                gain_curr = obj.marginalGain(set_, l_order[ord_id][i])
                #print(ev_curr, gainFast_curr, gain_curr)
                if math.isclose(round(ev_curr,3), round(ev_prev + gainFast_prev,3))==False:
                    flag[ord_id]=False        

                ev_prev = ev_curr
                gainFast_prev = gainFast_curr
                obj.sequentialUpdate(set_, l_order[ord_id][i])
                set_.add(l_order[ord_id][i])

            obj.clearPreCompute()
        assert flag[0]==flag[1] and flag[0]==True
        


