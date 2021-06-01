# facilityLocationVariantMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FacilityLocationVariantMutualInformation 
from submodlib.helper import create_kernel

class FacilityLocationVariantMutualInformationFunction(SetFunction):
	"""Implementation of the FacilityLocationVariantMutualInformation function.

	FacilityLocationVariantMutualInformation models diversity by computing the sum of pairwise distances of all the elements in a subset. It is defined as

	.. math::
			f(X) = \\sum_{i, j \\in X} (1 - s_{ij})

	Parameters
	----------

	n : int
		Number of elements in the ground set
	
	sijs : list, optional
		Similarity matrix to be used for getting :math:`s_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the similarity matrix

	metric : str, optional
		Similarity metric to be used for computing the similarity matrix
	
	n_neighbors : int, optional
		While constructing similarity matrix, number of nearest neighbors whose similarity values will be kept resulting in a sparse similarity matrix for computation speed up (at the cost of accuracy)
	
	"""

	def __init__(self, n, num_queries, query_sijs=None, imageData=None, queryData=None, metric="cosine", magnificationLambda=1):
		self.n = n
		self.num_queries = num_queries
		self.metric = metric
		self.query_sijs = query_sijs
		self.imageData = imageData
		self.queryData = queryData
		self.magnificationLambda=magnificationLambda
		self.cpp_obj = None
		self.cpp_query_sijs = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.num_queries < 0:
			raise Exception("ERROR: Number of queries must be >= 0")

		if self.metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if type(self.query_sijs) != type(None): # User has provided query kernel
			if type(self.query_sijs) != np.ndarray:
				raise Exception("Invalid query kernel type provided, must be ndarray")
			if np.shape(self.query_sijs)[0]!=self.n or np.shape(self.query_sijs)[1]!=self.num_queries:
				raise Exception("ERROR: Query Kernel should be n X num_queries")
			if (type(self.imageData) != type(None)) or (type(self.queryData) != type(None)):
				print("WARNING: similarity query kernel found. Provided image and query data matrices will be ignored.")
		else: #similarity query kernel has not been provided
			if (type(self.imageData) == type(None)) or (type(self.queryData) == type(None)):
				raise Exception("Since query kernel is not provided, data matrices are a must")
			if np.shape(self.imageData)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given image data matrix")
			if np.shape(self.queryData)[0]!=self.num_queries:
				raise Exception("ERROR: Inconsistentcy between num_queries and no of examples in the given query data matrix")
			
		    #construct queryKernel
			self.query_sijs = np.array(subcp.create_kernel_NS(self.queryData.tolist(),self.imageData.tolist(), self.metric))
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		self.cpp_query_sijs = self.query_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_query_sijs[0])==int or type(self.cpp_query_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_query_sijs)
			self.cpp_query_sijs=l

		self.cpp_obj = FacilityLocationVariantMutualInformation(self.n, self.num_queries, self.cpp_query_sijs, self.magnificationLambda)
		self.effective_ground = set(range(n))

	