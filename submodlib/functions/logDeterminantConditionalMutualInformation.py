# logDeterminantConditionalMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import LogDeterminantConditionalMutualInformation 
from submodlib.helper import create_kernel

class LogDeterminantConditionalMutualInformationFunction(SetFunction):
	"""Implementation of the LogDeterminantConditionalMutualInformation function.

	LogDeterminantConditionalMutualInformation models diversity by computing the sum of pairwise distances of all the elements in a subset. It is defined as

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

	def __init__(self, n, num_queries, num_privates, lambdaVal, image_sijs=None, query_sijs=None, query_query_sijs=None, private_sijs=None, private_private_sijs=None, query_private_sijs=None, imageData=None, queryData=None, privateData=None, metric="cosine", magnificationLambda=1, privacyHardness=1):
		self.n = n
		self.num_queries = num_queries
		self.num_privates = num_privates
		self.lambdaVal=lambdaVal
		self.metric = metric
		self.magnificationLambda=magnificationLambda
		self.privacyHardness=privacyHardness
		self.image_sijs = image_sijs
		self.query_sijs = query_sijs
		self.query_query_sijs = query_query_sijs
		self.private_sijs = private_sijs
		self.private_private_sijs = private_private_sijs
		self.query_private_sijs = query_private_sijs
		self.imageData = imageData
		self.queryData = queryData
		self.privateData = privateData
		
		self.cpp_obj = None
		self.cpp_image_sijs = None
		self.cpp_query_sijs = None
		self.cpp_query_query_sijs = None
		self.cpp_private_sijs = None
		self.cpp_private_private_sijs = None
		self.cpp_query_private_sijs = None
		self.cpp_content = None
		self.cpp_content2 = None
		self.cpp_content3 = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.num_queries < 0:
			raise Exception("ERROR: Number of queries must be >= 0")
		
		if self.num_privates < 0:
			raise Exception("ERROR: Number of queries must be >= 0")

		if self.metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if (type(self.image_sijs) != type(None)) and (type(self.query_sijs) != type(None)) and (type(self.query_query_sijs) != type(None)) and (type(self.private_sijs) != type(None)) and (type(self.private_private_sijs) != type(None)) and (type(self.query_private_sijs) != type(None)): # User has provided all required kernels
			if type(self.image_sijs) != np.ndarray:
				raise Exception("Invalid image kernel type provided, must be ndarray")
			if type(self.query_sijs) != np.ndarray:
				raise Exception("Invalid query kernel type provided, must be ndarray")
			if type(self.query_query_sijs) != np.ndarray:
				raise Exception("Invalid query-query kernel type provided, must be ndarray")
			if type(self.private_sijs) != np.ndarray:
				raise Exception("Invalid private kernel type provided, must be ndarray")
			if type(self.private_private_sijs) != np.ndarray:
				raise Exception("Invalid private-private kernel type provided, must be ndarray")
			if type(self.query_private_sijs) != np.ndarray:
				raise Exception("Invalid query-private kernel type provided, must be ndarray")
			if np.shape(self.image_sijs)[0]!=self.n or np.shape(self.image_sijs)[1]!=self.n:
				raise Exception("ERROR: Image Kernel should be n X n")
			if np.shape(self.query_sijs)[0]!=self.n or np.shape(self.query_sijs)[1]!=self.num_queries:
				raise Exception("ERROR: Query Kernel should be n X num_queries")
			if np.shape(self.query_query_sijs)[0]!=self.num_queries or np.shape(self.query_query_sijs)[1]!=self.num_queries:
				raise Exception("ERROR: Query-query Kernel should be num_queries X num_queries")
			if np.shape(self.private_sijs)[0]!=self.n or np.shape(self.private_sijs)[1]!=self.num_privates:
				raise Exception("ERROR: Private Kernel should be n X num_privates")
			if np.shape(self.private_private_sijs)[0]!=self.num_privates or np.shape(self.private_private_sijs)[1]!=self.num_privates:
				raise Exception("ERROR: Private-private Kernel should be num_privates X num_privates")
			if np.shape(self.query_private_sijs)[0]!=self.num_queries or np.shape(self.query_private_sijs)[1]!=self.num_privates:
				raise Exception("ERROR: Query-private Kernel should be num_queries X num_privates")
			if (type(self.imageData) != type(None)) or (type(self.queryData) != type(None)) or (type(self.privateData) != type(None)):
				print("WARNING: similarity kernels found. Provided image, query and private data matrices will be ignored.")
		else: #similarity kernels have not been provided
			if (type(self.imageData) == type(None)) or (type(self.queryData) == type(None)) or (type(self.privateData) == type(None)):
				raise Exception("Since kernels are not provided, data matrices are a must")
			if np.shape(self.imageData)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given image data matrix")
			if np.shape(self.queryData)[0]!=self.num_queries:
				raise Exception("ERROR: Inconsistentcy between num_queries and no of examples in the given query data matrix")
			if np.shape(self.privateData)[0]!=self.num_privates:
				raise Exception("ERROR: Inconsistentcy between num_privates and no of examples in the given private data matrix")
			
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

			#construct queryQueryKernel
			self.num_neighbors2 = self.num_queries #Using all data as num_neighbors in case of dense mode
			self.cpp_content2 = np.array(subcp.create_kernel(self.queryData.tolist(), self.metric, self.num_neighbors2))
			val2 = self.cpp_content2[0]
			row2 = list(self.cpp_content2[1].astype(int))
			col2 = list(self.cpp_content2[2].astype(int))
			self.query_query_sijs = np.zeros((self.num_queries,self.num_queries))
			self.query_query_sijs[row2,col2] = val2

			#construct privateKernel
			self.private_sijs = np.array(subcp.create_kernel_NS(self.privateData.tolist(),self.imageData.tolist(), self.metric))

			#construct privatePrivateKernel
			self.num_neighbors3 = self.num_privates #Using all data as num_neighbors in case of dense mode
			self.cpp_content3 = np.array(subcp.create_kernel(self.privateData.tolist(), self.metric, self.num_neighbors3))
			val3 = self.cpp_content3[0]
			row3 = list(self.cpp_content3[1].astype(int))
			col3 = list(self.cpp_content3[2].astype(int))
			self.private_private_sijs = np.zeros((self.num_privates,self.num_privates))
			self.private_private_sijs[row3,col3] = val3

			#construct queryPrivateKernel
			self.query_private_sijs = np.array(subcp.create_kernel_NS(self.privateData.tolist(),self.queryData.tolist(), self.metric))
		
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
		
		self.cpp_query_query_sijs = self.query_query_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_query_query_sijs[0])==int or type(self.cpp_query_query_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_query_query_sijs)
			self.cpp_query_query_sijs=l
		
		self.cpp_private_sijs = self.private_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_private_sijs[0])==int or type(self.cpp_private_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_private_sijs)
			self.cpp_private_sijs=l
		
		self.cpp_private_private_sijs = self.private_private_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_private_private_sijs[0])==int or type(self.cpp_private_private_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_private_private_sijs)
			self.cpp_private_private_sijs=l
		
		self.cpp_query_private_sijs = self.query_private_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_query_private_sijs[0])==int or type(self.cpp_query_private_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_query_private_sijs)
			self.cpp_query_private_sijs=l

		self.cpp_obj = LogDeterminantConditionalMutualInformation(self.n, self.num_queries, self.num_privates, self.cpp_image_sijs, self.cpp_query_sijs, self.cpp_query_query_sijs, self.cpp_private_sijs, self.cpp_private_private_sijs,self.cpp_query_private_sijs, self.lambdaVal, self.magnificationLambda, self.privacyHardness)
		self.effective_ground = set(range(n))

	def evaluate(self, X):
		"""Computes the LogDeterminantConditionalMutualInformation score of a set

		Parameters
		----------
		X : set
			The set whose LogDeterminantConditionalMutualInformation score needs to be computed
		
		Returns
		-------
		float
			The LogDeterminantConditionalMutualInformation function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False):
		"""Find the optimal subset with maximum LogDeterminantConditionalMutualInformation score for a given budget

		Parameters
		----------
		budget : int
			Desired size of the optimal set
		optimizer : string
			The optimizer that should be used to compute the optimal set. Can be 'NaiveGreedy', 'LazyGreedy', 'LazierThanLazyGreedy'
		stopIfZeroGain : bool
			Set to True if budget should be filled with items adding zero gain. If False, size of optimal set can be potentially less than the budget
		stopIfNegativeGain : bool
			Set to True if maximization should terminate as soon as the best gain in an iteration is negative. This can potentially lead to optimal set of size less than the budget
		verbose : bool
			Set to True to trace the execution of the maximization algorithm

		Returns
		-------
		set
			The optimal set of size budget

		"""

		if budget >= len(self.effective_ground):
			raise Exception(f"Budget {budget} must be less than effective ground set size {len(self.effective_ground)}")
		return self.cpp_obj.maximize(optimizer, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose)
	
	def marginalGain(self, X, element):
		"""Find the marginal gain in LogDeterminantConditionalMutualInformation score when a single item (element) is added to a set (X)

		Parameters
		----------
		X : set
			Set on which the marginal gain of adding an element has to be calculated. It must be a subset of the effective ground set.
		element : int
			Element for which the marginal gain is to be calculated. It must be from the effective ground set.

		Returns
		-------
		float
			Marginal gain of adding element to X

		"""

		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if type(element)!=int:
			raise Exception("ERROR: element should be an int")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X is not a subset of effective ground set")

		if element not in self.effective_ground:
			raise Exception("Error: element must be in the effective ground set")

		if element in X:
			return 0

		return self.cpp_obj.marginalGain(X, element)

	def marginalGainWithMemoization(self, X, element):
		"""Efficiently find the marginal gain in LogDeterminantConditionalMutualInformation score when a single item (element) is added to a set (X) assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			Set on which the marginal gain of adding an element has to be calculated. It must be a subset of the effective ground set and its memoized statistics should have already been computed
		element : int
			Element for which the marginal gain is to be calculated. It must be from the effective ground set.

		Returns
		-------
		float
			Marginal gain of adding element to X

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if type(element)!=int:
			raise Exception("ERROR: element should be an int")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X is not a subset of effective ground set")

		if element not in self.effective_ground:
			raise Exception("Error: element must be in the effective ground set")

		if element in X:
			return 0

		return self.cpp_obj.marginalGainWithMemoization(X, element)

	def evaluateWithMemoization(self, X):
		"""Efficiently compute the LogDeterminantConditionalMutualInformation score of a set assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			The set whose LogDeterminantConditionalMutualInformation score needs to be computed
		
		Returns
		-------
		float
			The LogDeterminantConditionalMutualInformation function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0

		return self.cpp_obj.evaluateWithMemoization(X)

	def updateMemoization(self, X, element):
		"""Update the memoized statistics of X due to adding element to X. Assumes that memoized statistics are already computed for X

		Parameters
		----------
		X : set
			Set whose memoized statistics must already be computed and to which the element needs to be added for the sake of updating the memoized statistics
		element : int
			Element that is being added to X leading to update of memoized statistics. It must be from effective ground set.

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if type(element)!=int:
			raise Exception("ERROR: element should be an int")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X is not a subset of effective ground set")

		if element not in self.effective_ground:
			raise Exception("Error: element must be in the effective ground set")

		if element in X:
			return

		self.cpp_obj.updateMemoization(X, element)
	
	def clearMemoization(self):
		"""Clear the computed memoized statistics, if any

		"""
		self.cpp_obj.clearMemoization()
	
	def setMemoization(self, X):
		"""Compute and store the memoized statistics for subset X 

		Parameters
		----------
		X : set
			The set for which memoized statistics need to be computed and set
		
		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		self.cpp_obj.setMemoization(X)
	
	def getEffectiveGroundSet(self):
		"""Get the effective ground set of this LogDeterminantConditionalMutualInformation object. This is equal to the ground set when instantiated with partial=False and is equal to the ground_sub when instantiated with partial=True

		"""
		return self.cpp_obj.getEffectiveGroundSet()