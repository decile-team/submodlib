# graphCutConditionalGain.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import GraphCutConditionalGain 
from submodlib.helper import create_kernel

class GraphCutConditionalGainFunction(SetFunction):
	"""Implementation of the Graph Cut Conditional Gain (GCCG) function.

	Given a :ref:`functions.conditional-gain` function, Graph Cut Conditional Gain function is its instantiation using a :class:`~submodlib.functions.graphCut.GraphCutFunction`. Mathematically, it takes the following form:

	.. math::
			f(A | P) = f_{\\lambda}(A) - 2 \\lambda \\nu \\sum\\limits_{i \\in A, j \\in P} s_{ij}

	Where :math:`\\nu` is an additional parameter that controls the hardness of the privacy constraint and
	
	.. math::
			f_{\\lambda}(A) = \\sum_{i \\in V, j \\in A} s_{ij} - \\lambda \\sum_{i, j \\in A} s_{ij}

	.. note::
			The submodular function used by :cite:`li2012multi` in update-summarization is GCCG.
	
	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	num_privates : int
		Number of private instances in the target.

	lambdaVal : float
		The representation and diversity trade-off parameter :math:`\\lambda` in :class:`~submodlib.functions.graphCut.GraphCutFunction`
	
	data_sijs : numpy.ndarray, optional
		Similarity kernel between the elements of the ground set. Shape: n X n. When not provided, it is computed using data.
	
	private_sijs : numpy.ndarray, optional
		Similarity kernel between the ground set and the private instances. Shape: n X num_privates. When not provided, it is computed using data and privateData.

	data : numpy.ndarray, optional
		Matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. Mandatory, if either if data_sijs or private_sijs is not provided. Ignored if both data_sijs and private_sijs are provided.

	privateData : numpy.ndarray, optional
		Matrix of shape num_privates X num_features containing the private instances. privateData[i] should contain the num-features dimensional features of private instance i. Must be provided if private_sijs is not provided. Ignored if both data_sijs and private_sijs are provided.

	metric : str, optional
		Similarity metric to be used for computing the similarity kernels. Can be "cosine" for cosine similarity or "euclidean" for similarity based on euclidean distance. Default is "cosine". 

	privacyHardness : float, optional
		Parameter that governs the hardness of the privacy constraint. Default is 1.
	
	"""

	def __init__(self, n, num_privates, lambdaVal, data_sijs=None, private_sijs=None, data=None, privateData=None, metric="cosine", privacyHardness=1):
		self.n = n
		self.num_privates = num_privates
		self.lambdaVal =lambdaVal
		self.metric = metric
		self.data_sijs = data_sijs
		self.private_sijs = private_sijs
		self.data = data
		self.privateData = privateData
		self.privacyHardness=privacyHardness
		self.cpp_obj = None
		self.cpp_data_sijs = None
		self.cpp_private_sijs = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.num_privates < 0:
			raise Exception("ERROR: Number of queries must be >= 0")

		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if (type(self.data_sijs) != type(None)) and (type(self.private_sijs) != type(None)): # User has provided both kernels
			if type(self.data_sijs) != np.ndarray:
				raise Exception("Invalid data kernel type provided, must be ndarray")
			if type(self.private_sijs) != np.ndarray:
				raise Exception("Invalid query kernel type provided, must be ndarray")
			if np.shape(self.data_sijs)[0]!=self.n or np.shape(self.data_sijs)[1]!=self.n:
				raise Exception("ERROR: data kernel should be n X n")
			if np.shape(self.private_sijs)[0]!=self.n or np.shape(self.private_sijs)[1]!=self.num_privates:
				raise Exception("ERROR: Query Kernel should be n X num_privates")
			if (type(self.data) != type(None)) or (type(self.privateData) != type(None)):
				print("WARNING: similarity kernels found. Provided data and query matrices will be ignored.")
		else: #similarity kernels have not been provided
			if (type(self.data) == type(None)) or (type(self.privateData) == type(None)):
				raise Exception("Since kernels are not provided, data matrices are a must")
			if np.shape(self.data)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
			if np.shape(self.privateData)[0]!=self.num_privates:
				raise Exception("ERROR: Inconsistentcy between num_privates and no of examples in the given query data matrix")
			
			#construct imageKernel
			self.num_neighbors = self.n #Using all data as num_neighbors in case of dense mode
			self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
			val = self.cpp_content[0]
			row = list(self.cpp_content[1].astype(int))
			col = list(self.cpp_content[2].astype(int))
			self.data_sijs = np.zeros((self.n,self.n))
			self.data_sijs[row,col] = val
		
		    #construct privateKernel
			self.private_sijs = np.array(subcp.create_kernel_NS(self.privateData.tolist(),self.data.tolist(), self.metric))
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		self.cpp_data_sijs = self.data_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_data_sijs[0])==int or type(self.cpp_data_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_data_sijs)
			self.cpp_data_sijs=l
		
		self.cpp_private_sijs = self.private_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_private_sijs[0])==int or type(self.cpp_private_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_private_sijs)
			self.cpp_private_sijs=l

		self.cpp_obj = GraphCutConditionalGain(self.n, self.num_privates, self.cpp_data_sijs, self.cpp_private_sijs, self.privacyHardness, self.lambdaVal)
		self.effective_ground = set(range(n))