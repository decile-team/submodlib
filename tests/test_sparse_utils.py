import pytest
import math
import numpy as np
from scipy import sparse
from submodlib_cpp import SparseSim

X = np.array([
[0.01,0.2,0.43,0.04,-0.5],
[0,0,0,0,0],
[1,-1,-1,1,-1],
[0,0,0.57,0,0],
[0.75,-0.6,0,0.65,-0.57]])

X_csr = sparse.csr_matrix(X)

arr_val = X_csr.data.tolist()
arr_count = X_csr.indptr.tolist()
arr_col = X_csr.indices.tolist()

obj = SparseSim(arr_val, arr_count, arr_col)

class TestSparseUtils:

    def test_val1(self):
        assert math.isclose(round(obj.get_val(0,0),2), 0.01)
    
    def test_val2(self):
        assert math.isclose(round(obj.get_val(0,2),2), 0.43)

    def test_val3(self):
        assert math.isclose(round(obj.get_val(2,2),2), -1)
    
    def test_row1(self):
        assert np.allclose(obj.get_row(0), X[0,:])

    def test_row2(self):
        assert np.allclose(obj.get_row(1), X[1,:])

    def test_row3(self):
        assert np.allclose(obj.get_row(4), X[4,:])

    def test_col1(self):
        assert np.allclose(obj.get_col(0), X[:,0])

    def test_col2(self):
        assert np.allclose(obj.get_col(1), X[:,1])

    def test_col3(self):
        assert np.allclose(obj.get_col(4), X[:,4])

