# logDeterminant.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from scipy import sparse
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import LogDeterminant 
from submodlib.helper import create_kernel, create_cluster_kernels


class LogDeterminantFunction(SetFunction):
	"""Implementation of the Log-Determinant (LogDet) function.

	Given a positive semi-definite kernel matrix :math:`L`, denote :math:`L_A` as the subset of rows and columns indexed by the set :math:`A`. The log-determinant function is 
	
	.. math::
			f(A) = \\log\\det(L_A)
	
	The log-det function models diversity, and is closely related to a determinantal point process :cite:`kulesza2012determinantal`.
	
	Determinantal Point Processes (DPP) are an example of functions that model diversity. DPPs are defined as
	
	.. math::
			p(X) = \\mbox{Det}(S_X)
	
	where :math:`S` is a similarity kernel matrix, and :math:`S_X` denotes the rows and columns instantiated with elements in :math:`X`. It turns out that :math:`f(X) = \\log p(X)` is submodular, and hence can be efficiently optimized via the greedy algorithm.	

	.. note::
			DPP requires computing the determinant and is :math:`\\mathcal{O}(n^3)` where :math:`n` is the size of the ground set.
	
	.. note::
			The implementation follows Fast Greedy MAP Inference as presented in :cite:`chen2018fast`

	Parameters
	----------
	n : int
		Number of elements in the ground set, must be > 0.
	
	mode : string
		Can be "dense" or "sparse". It specifies whether the Log Determinant function should operate in dense mode (using a dense similarity kernel) or sparse mode (using a sparse similarity kernel).

	lambdaVal : float
		Value to add to :math:`s_{ii} (i.e. 1)` so that :math:`\\log` doesn't become 0.
	
	sijs : numpy.ndarray or scipy.sparse.csr.csr_matrix, optional
		Similarity kernel (dense or sparse) between the elements of the ground set, to be used for getting :math:`L` as defined above. Shape of dense kernel must be n X n. When not provided, it is computed internally in C++ based on the following additional parameters.
	
	data : numpy.ndarray, optional
		Matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. Used to compute the similarity kernel. It is optional (and is ignored if provided) if sijs has been provided.

	metric : str, optional
		Similarity metric to be used for computing the similarity kernel. Can be "cosine" for cosine similarity or "euclidean" for similarity based on euclidean distance. Default is "cosine".
	
	num_neighbors : int, optional
		Number of neighbors applicable for the sparse similarity kernel. Must not be provided if mode is "dense". Must be provided if either a sparse kernel is provided or is to be computed.
	
	"""

	def __init__(self, n, mode, lambdaVal, sijs=None, data=None, metric="cosine", num_neighbors=None):
		self.n = n
		self.mode = mode
		self.metric = metric
		self.sijs = sijs
		self.data = data
		self.num_neighbors = num_neighbors
		self.lambdaVal = lambdaVal
		self.cpp_obj = None
		self.cpp_sijs = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in ['dense', 'sparse', 'clustered']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'dense', 'sparse' or 'clustered'")
		
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

			self.cpp_obj = LogDeterminant(self.n, self.cpp_sijs, False, cpp_ground_sub, self.lambdaVal)
		
		if self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = LogDeterminant(self.n, self.cpp_sijs['arr_val'], self.cpp_sijs['arr_count'], self.cpp_sijs['arr_col'], self.lambdaVal)
		
		self.effective_ground = self.cpp_obj.getEffectiveGroundSet()