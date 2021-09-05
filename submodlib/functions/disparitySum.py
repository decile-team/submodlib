# disparitySum.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from scipy import sparse
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import DisparitySum 
#from submodlib.helper import create_kernel, create_cluster_kernels

class DisparitySumFunction(SetFunction):
	"""Implementation of the Disparity-Sum (DispSum) function.

	Disparity-Sum models diversity by computing the sum of pairwise distances of all the elements in a subset. It is defined as

	.. math::
			f(X) = \\sum_{i, j \\in X} (1 - s_{ij})

	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	mode : str
		Can be "dense" or "sparse". It specifies whether the Disparity-Sum function should operate in dense mode (using a dense similarity kernel) or sparse mode (using a sparse similarity kernel).
	
	sijs : numpy.ndarray or scipy.sparse.csr.csr_matrix, optional
		Similarity kernel (dense or sparse) between the elements of the ground set, to be used for getting :math:`s_{ij}` entries as defined above. Shape of dense kernel must be n X n. When not provided, it is computed internally in C++ based on the following additional parameters. **The implementation requires this simimlarity kernel to be normalized, i.e. entries must be strictly in [0,1].**

	data : numpy.ndarray, optional
		Matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. Used to compute the similarity kernel. It is optional (and is ignored if provided) if sijs has been provided.

	metric : str, optional
		Similarity metric to be used for computing the similarity kernel. Can be "cosine" for cosine similarity or "euclidean" for similarity based on euclidean distance. Default is "cosine".
	
	num_neighbors : int, optional
		Number of neighbors applicable for the sparse similarity kernel. Must not be provided if mode is "dense". Must be provided if either a sparse kernel is provided or is to be computed.
	
	"""

	def __init__(self, n, mode, sijs=None, data=None, metric="cosine", num_neighbors=None):
		self.n = n
		self.mode = mode
		self.metric = metric
		self.sijs = sijs
		self.data = data
		self.num_neighbors = num_neighbors
		self.cpp_obj = None
		self.cpp_sijs = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in ['dense', 'sparse']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")
		
		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if type(self.sijs) != type(None): # User has provided similarity kernel
			if type(self.sijs) == scipy.sparse.csr.csr_matrix:
				if num_neighbors is None or num_neighbors <= 0:
					raise Exception("ERROR: Positive num_neighbors must be provided for given sparse kernel")
				if mode != "sparse":
					raise Exception("ERROR: Sparse kernel provided, but mode is not sparse")
			elif type(self.sijs) == np.ndarray:
				if mode != "dense":
					raise Exception("ERROR: Dense kernel provided, but mode is not dense")
			else:
				raise Exception("Invalid kernel provided")
			#TODO: is the below dimensionality check valid for both dense and sparse kernels?
			if np.shape(self.sijs)[0]!=self.n or np.shape(self.sijs)[1]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity kernel")
			if type(self.data) != type(None):
				print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
		else: #similarity kernel has not been provided
			if type(self.data) != type(None): 
				if np.shape(self.data)[0]!=self.n:
					raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
				
				if self.mode == "dense":
					if self.num_neighbors  is not None:
						raise Exception("num_neighbors wrongly provided for dense mode")
					self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
				self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
				val = self.cpp_content[0]
				row = list(self.cpp_content[1].astype(int))
				col = list(self.cpp_content[2].astype(int))
				if self.mode=="dense":
					self.sijs = np.zeros((n,n))
					self.sijs[row,col] = val
				if self.mode=="sparse":
					self.sijs = sparse.csr_matrix((val, (row, col)), [n,n])
			else:
				raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
		
		cpp_ground_sub = {-1} #Provide a dummy set for pybind11 binding to be successful
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		if self.mode=="dense":
			self.cpp_sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure
			
			if type(self.cpp_sijs[0])==int or type(self.cpp_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			 #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(self.cpp_sijs)
				self.cpp_sijs=l

			self.cpp_obj = DisparitySum(self.n, self.cpp_sijs, False, cpp_ground_sub)
		
		if self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = DisparitySum(self.n, self.cpp_sijs['arr_val'], self.cpp_sijs['arr_count'], self.cpp_sijs['arr_col'])
		
		self.effective_ground = self.cpp_obj.getEffectiveGroundSet()