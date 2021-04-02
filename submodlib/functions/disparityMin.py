# disparityMin.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from scipy import sparse
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import DisparityMin 
from submodlib.helper import create_kernel, create_cluster_kernels


class DisparityMinFunction(SetFunction):
	"""Implementation of the Disparity-Min function.
	
	Diversity based functions attempt to obtain a diverse set of keypoints. The goal is to have minimum similarity across elements in the chosen subset by maximizing minimum pairwise distance between elements. There is a subtle difference between the notion of diversity and the notion of representativeness. While diversity *only* looks at the elements in the chosen subset, representativeness also worries about their similarity with the remaining elements in the superset. Denote :math:`d_{ij}` as a distance measure between images/data points :math:`i` and :math:`j`. Disparity-Min for a subset :math:`X` is defined as: 
	
	.. math::
			f(X) = \\min_{i, j \\in X, i \\neq j} d_{ij}

	It is easy to see that maximizing this function involves obtaining a subset with maximal minimum pairwise distance, thereby ensuring a diverse subset of datapoints (snippets or keyframes) in the summary.

	.. note::
			This function is not submodular, but can be efficiently optimized via a greedy algorithm :cite:`dasgupta2013summarization`.

	Parameters
	----------

	n : int
		Number of elements in the ground set
	
	dijs : list, optional
		Distance matrix to be used for getting :math:`d_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the distance matrix

	metric : str, optional
		Distance metric to be used for computing the distance matrix
	
	n_neighbors : int, optional
		While constructing distance matrix, number of nearest neighbors whose distance values will be kept resulting in a sparse distance matrix for computation speed up (at the cost of accuracy)

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

		if self.mode not in ['dense', 'sparse', 'clustered']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'dense', 'sparse' or 'clustered'")
		
		if self.metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

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

			self.cpp_obj = DisparityMin(self.n, self.cpp_sijs, False, cpp_ground_sub)
		
		if self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = DisparityMin(self.n, self.cpp_sijs['arr_val'], self.cpp_sijs['arr_count'], self.cpp_sijs['arr_col'])
		
		self.effective_ground = self.cpp_obj.getEffectiveGroundSet()

	def evaluate(self, X):
		"""Computes the Disparity-Min score of a set

		Parameters
		----------
		X : set
			The set whose Disparity-Min score needs to be computed
		
		Returns
		-------
		float
			The Disparity-Min function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False):
		"""Find the optimal subset with maximum Disparity-Min score for a given budget

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
			raise Exception("Budget must be less than effective ground set size")
		return self.cpp_obj.maximize(optimizer, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose)
	
	def marginalGain(self, X, element):
		"""Find the marginal gain in Disparity-Min score when a single item (element) is added to a set (X)

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
		"""Efficiently find the marginal gain in Disparity-Min score when a single item (element) is added to a set (X) assuming that memoized statistics for it are already computed

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
		"""Efficiently compute the Disparity-Min score of a set assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			The set whose Disparity-Min score needs to be computed
		
		Returns
		-------
		float
			The Disparity-Min function evaluation on the given set

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
		"""Get the effective ground set of this Disparity-Min object. This is equal to the ground set when instantiated with partial=False and is equal to the ground_sub when instantiated with partial=True

		"""
		return self.cpp_obj.getEffectiveGroundSet()