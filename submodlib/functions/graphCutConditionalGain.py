# graphCutConditionalGain.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import GraphCutConditionalGain 
from submodlib.helper import create_kernel

class GraphCutConditionalGainFunction(SetFunction):
	"""Implementation of the GraphCutConditionalGain function.

	GraphCutConditionalGain models diversity by computing the sum of pairwise distances of all the elements in a subset. It is defined as

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

	def __init__(self, n, num_privates, lambdaVal, image_sijs=None, private_sijs=None, imageData=None, privateData=None, metric="cosine", privacyHardness=1):
		self.n = n
		self.num_privates = num_privates
		self.lambdaVal =lambdaVal
		self.metric = metric
		self.image_sijs = image_sijs
		self.private_sijs = private_sijs
		self.imageData = imageData
		self.privateData = privateData
		self.privacyHardness=privacyHardness
		self.cpp_obj = None
		self.cpp_image_sijs = None
		self.cpp_private_sijs = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.num_privates < 0:
			raise Exception("ERROR: Number of queries must be >= 0")

		if self.metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if (type(self.image_sijs) != type(None)) and (type(self.private_sijs) != type(None)): # User has provided both kernels
			if type(self.image_sijs) != np.ndarray:
				raise Exception("Invalid image kernel type provided, must be ndarray")
			if type(self.private_sijs) != np.ndarray:
				raise Exception("Invalid query kernel type provided, must be ndarray")
			if np.shape(self.image_sijs)[0]!=self.n or np.shape(self.image_sijs)[1]!=self.n:
				raise Exception("ERROR: Image Kernel should be n X n")
			if np.shape(self.private_sijs)[0]!=self.n or np.shape(self.private_sijs)[1]!=self.num_privates:
				raise Exception("ERROR: Query Kernel should be n X num_privates")
			if (type(self.imageData) != type(None)) or (type(self.privateData) != type(None)):
				print("WARNING: similarity kernels found. Provided image and query data matrices will be ignored.")
		else: #similarity kernels have not been provided
			if (type(self.imageData) == type(None)) or (type(self.privateData) == type(None)):
				raise Exception("Since kernels are not provided, data matrices are a must")
			if np.shape(self.imageData)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given image data matrix")
			if np.shape(self.privateData)[0]!=self.num_privates:
				raise Exception("ERROR: Inconsistentcy between num_privates and no of examples in the given query data matrix")
			
			#construct imageKernel
			self.num_neighbors = self.n #Using all data as num_neighbors in case of dense mode
			self.cpp_content = np.array(subcp.create_kernel(self.imageData.tolist(), self.metric, self.num_neighbors))
			val = self.cpp_content[0]
			row = list(self.cpp_content[1].astype(int))
			col = list(self.cpp_content[2].astype(int))
			self.image_sijs = np.zeros((self.n,self.n))
			self.image_sijs[row,col] = val
		
		    #construct privateKernel
			self.private_sijs = np.array(subcp.create_kernel_NS(self.privateData.tolist(),self.imageData.tolist(), self.metric))
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		self.cpp_image_sijs = self.image_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_image_sijs[0])==int or type(self.cpp_image_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_image_sijs)
			self.cpp_image_sijs=l
		
		self.cpp_private_sijs = self.private_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_private_sijs[0])==int or type(self.cpp_private_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_private_sijs)
			self.cpp_private_sijs=l

		self.cpp_obj = GraphCutConditionalGain(self.n, self.num_privates, self.cpp_image_sijs, self.cpp_private_sijs, self.privacyHardness, self.lambdaVal)
		self.effective_ground = set(range(n))

	