# facilityLocationMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FacilityLocationMutualInformation 
from submodlib.helper import create_kernel

class FacilityLocationMutualInformationFunction(SetFunction):
	"""Implementation of the FacilityLocationMutualInformation function.

	FacilityLocationMutualInformation models diversity by computing the sum of pairwise distances of all the elements in a subset. It is defined as

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

	def __init__(self, n, num_queries, image_sijs=None, query_sijs=None, imageData=None, queryData=None, metric="cosine", magnificationLambda=1):
		self.n = n
		self.num_queries = num_queries
		self.metric = metric
		self.image_sijs = image_sijs
		self.query_sijs = query_sijs
		self.imageData = imageData
		self.queryData = queryData
		self.magnificationLambda=magnificationLambda
		self.cpp_obj = None
		self.cpp_image_sijs = None
		self.cpp_query_sijs = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.num_queries < 0:
			raise Exception("ERROR: Number of queries must be >= 0")

		if self.metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if (type(self.image_sijs) != type(None)) and (type(self.query_sijs) != type(None)): # User has provided both kernels
			if type(self.image_sijs) != np.ndarray:
				raise Exception("Invalid image kernel type provided, must be ndarray")
			if type(self.query_sijs) != np.ndarray:
				raise Exception("Invalid query kernel type provided, must be ndarray")
			if np.shape(self.image_sijs)[0]!=self.n or np.shape(self.image_sijs)[1]!=self.n:
				raise Exception("ERROR: Image Kernel should be n X n")
			if np.shape(self.query_sijs)[0]!=self.n or np.shape(self.query_sijs)[1]!=self.num_queries:
				raise Exception("ERROR: Query Kernel should be n X num_queries")
			if (type(self.imageData) != type(None)) or (type(self.queryData) != type(None)):
				print("WARNING: similarity kernels found. Provided image and query data matrices will be ignored.")
		else: #similarity kernels have not been provided
			if (type(self.imageData) == type(None)) or (type(self.queryData) == type(None)):
				raise Exception("Since kernels are not provided, data matrices are a must")
			if np.shape(self.imageData)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given image data matrix")
			if np.shape(self.queryData)[0]!=self.num_queries:
				raise Exception("ERROR: Inconsistentcy between num_queries and no of examples in the given query data matrix")
			
			#construct imageKernel
			self.num_neighbors = self.n #Using all data as num_neighbors in case of dense mode
			self.cpp_content = np.array(subcp.create_kernel(self.imageData.tolist(), self.metric, self.num_neighbors))
			val = self.cpp_content[0]
			row = list(self.cpp_content[1].astype(int))
			col = list(self.cpp_content[2].astype(int))
			self.image_sijs = np.zeros((self.n,self.n))
			self.image_sijs[row,col] = val
		
		    #construct queryKernel
			self.query_sijs = np.array(subcp.create_kernel_NS(self.queryData.tolist(),self.imageData.tolist(), self.metric))
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		self.cpp_image_sijs = self.image_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_image_sijs[0])==int or type(self.cpp_image_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_image_sijs)
			self.cpp_image_sijs=l
		
		self.cpp_query_sijs = self.query_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_query_sijs[0])==int or type(self.cpp_query_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_query_sijs)
			self.cpp_query_sijs=l

		self.cpp_obj = FacilityLocationMutualInformation(self.n, self.num_queries, self.cpp_image_sijs, self.cpp_query_sijs, self.magnificationLambda)
		self.effective_ground = set(range(n))

	