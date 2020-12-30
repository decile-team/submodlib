import pytest
import math
import numpy as np
from scipy import sparse
from submodlib_cpp import SparseSim
'''
X = np.array([
[0.01,0.2,0.43,0.04,-0.5],
[0,0,0,0,0],
[1,-1,-1,1,-1],
[0,0,0.57,0,0],
[0.75,-0.6,0,0.65,-0.57]])
'''


X = np.array([
 [1.00000000e+00, 4.68004632e-02, 1.53625389e-02, 0, 0],
 [4.68004632e-02, 1.00000000e+00, 2.48592296e-01, 0, 0],
 [1.53625389e-02, 2.48592296e-01, 1.00000000e+00, 0, 0],
 [3.54753001e-23, 0, 0, 1.00000000e+00, 1.00000000e+00],
 [3.54753001e-23, 0, 0, 1.00000000e+00, 1.00000000e+00]])



X_csr = sparse.csr_matrix(X)

arr_val = X_csr.data.tolist()
arr_count = X_csr.indptr.tolist()
arr_col = X_csr.indices.tolist()

obj1 = SparseSim(arr_val, arr_count, arr_col)

obj2 = SparseSim(arr_val, arr_col, 3, 5)

class TestSparseUtils:
    
    #Test cases for obj1
    def test_val1(self):
        assert math.isclose(obj1.get_val(0,0), 1)
    
    def test_val2(self):
        assert math.isclose(round(obj1.get_val(0,2),4), 0.0154)

    def test_val3(self):
        assert math.isclose(obj1.get_val(2,3), 0)
    
    def test_row1(self):
        assert np.allclose(obj1.get_row(0), X[0,:])

    def test_row2(self):
        assert np.allclose(obj1.get_row(1), X[1,:])

    def test_row3(self):
        assert np.allclose(obj1.get_row(4), X[4,:])

    def test_col1(self):
        assert np.allclose(obj1.get_col(0), X[:,0])

    def test_col2(self):
        assert np.allclose(obj1.get_col(1), X[:,1])

    def test_col3(self):
        assert np.allclose(obj1.get_col(4), X[:,4])

    


    #Test cases for obj2
    def test_val4(self):
        assert math.isclose(obj2.get_val(0,0), 1)
    
    def test_val5(self):
        assert math.isclose(round(obj2.get_val(0,2),4), 0.0154)

    def test_val6(self):
        assert math.isclose(obj2.get_val(2,3), 0)
    
    def test_row4(self):
        assert np.allclose(obj2.get_row(0), X[0,:])

    def test_row5(self):
        assert np.allclose(obj2.get_row(1), X[1,:])

    def test_row6(self):
        assert np.allclose(obj2.get_row(4), X[4,:])

    def test_col4(self):
        assert np.allclose(obj2.get_col(0), X[:,0])

    def test_col5(self):
        assert np.allclose(obj2.get_col(1), X[:,1])

    def test_col6(self):
        assert np.allclose(obj2.get_col(4), X[:,4])

