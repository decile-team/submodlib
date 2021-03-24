# clustered.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
from .setFunction import SetFunction
from submodlib.helper import create_cluster_kernels
import submodlib_cpp as subcp
from submodlib_cpp import Clustered 

class ClusteredFunction(SetFunction):
	"""Implementation of the Clustered function.

	Given a function and a clustering, clustered function internally creates a mixture of functions each defined over a cluster. It is defined as
	
	.. math::
			f(X) = \\sum_i f_{C_i}(X)

	where :math:`f_{C_i}` operates only on cluster :math:`C_i` as sub-groundset and interprets :math:`X` as :math:`X \cap C_i`
	
	.. note::
			When the clusters are labels, this becomes supervised subset selection.

	Parameters
	----------
	n : int
		Number of elements in the ground set
	f_name : string
		Name of particular set function whose clustered implementation is desired 
	data : numpy ndarray
		Data matrix which will be used for computing the similarity kernel (and for computing the clusters if clustering is not provided)
	mode : string
		Can be "single" (to create a single dense large similarity kernel) or "multi" (to create one small dense kernel per cluster)
	cluster_lab : list, optional
		Its a list that contains cluster labels for each data point. If not provided, clustering is done internally using sklearn's BIRCH using provided data
	num_clusters : int, optional
		Number of clusters. Mandatory if cluster_lab is provided. If cluster_lab is not provided, clustering is done internally. In this case if num_clusters is not provided, an optimal number of clusters is created. 
	metric : string, optional
		similarity metric to be used while computing similarity kernel for each cluster or a dense kernel. Can be "euclidean" or "cosine". By default, it is "cosine"
		
	"""

	def __init__(self, n, f_name, data, mode, cluster_lab=None, num_clusters=None, metric="cosine", lambdaVal=1):
		self.n = n
		self.f_name = f_name
		self.num_clusters=num_clusters
		self.data = data
		self.mode = mode
		self.cluster_lab=cluster_lab
		self.metric = metric
		
		self.clusters=None
		self.cluster_sijs=None
		self.cluster_map=None
		self.sijs = None
		self.cpp_content = None
		self.cpp_sijs = None
		self.effective_ground=None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in ['single', 'multi']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'single' or 'multi'")

		if self.metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric")

		if type(self.cluster_lab) != type(None) and (self.num_clusters  is None or self.num_clusters <= 0):
			raise Exception("ERROR: Positive number of clusters must be provided when cluster_lab is provided")
		if type(self.cluster_lab) != type(None) and len(self.cluster_lab) != self.n:
			raise Exception("ERROR: cluster_lab's size is NOT same as ground set size")
		if type(self.cluster_lab) != type(None) and not all(ele >= 0 and ele <= self.num_clusters-1 for ele in self.cluster_lab):
			raise Exception("Cluster IDs/labels contain invalid values")

		if np.shape(self.data)[0]!=self.n:
			raise Exception("ERROR: Inconsistentcy between n and no of examples in the given ground data matrix")

		if mode == "single":
			self.clusters, _, _ = create_cluster_kernels(self.data.tolist(), self.metric, self.cluster_lab, self.num_clusters, onlyClusters=True)
			self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
			self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
			val = self.cpp_content[0]
			row = list(map(lambda arg: int(arg), self.cpp_content[1]))
			col = list(map(lambda arg: int(arg), self.cpp_content[2]))
			self.sijs = np.zeros((n,n))
			self.sijs[row,col] = val
			self.cpp_sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure
			if type(self.cpp_sijs[0])==int or type(self.cpp_sijs[0])==float: #Its critical that we pass a list of list to pybind11
			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(self.cpp_sijs)
				self.cpp_sijs=l
			self.cpp_obj = Clustered(self.n, self.f_name, self.clusters, self.cpp_sijs, lambdaVal)
		else:
			self.clusters, self.cluster_sijs, self.cluster_map = create_cluster_kernels(self.data.tolist(), self.metric, self.cluster_lab, self.num_clusters)
			l_temp = []
			#TODO: this for loop can be optimized
			for el in self.cluster_sijs:
				temp=el.tolist()
				if type(temp[0])==int or type(temp[0])==float: #Its critical that we pass a list of list to pybind11
																#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
					l=[]
					l.append(temp)
					temp=l
				l_temp.append(temp)
			self.cluster_sijs = l_temp
			#print("Clusters: ", self.clusters)
			#print("cluster_sijs: ", self.cluster_sijs)
			#print("cluster_map: ", self.cluster_map)
			self.cpp_obj = Clustered(self.n, self.f_name, self.clusters, self.cluster_sijs, self.cluster_map, lambdaVal)
		self.effective_ground=self.cpp_obj.getEffectiveGroundSet()

	def evaluate(self, X):
		"""Computes the Clustered Function score of a set

		Parameters
		----------
		X : set
			The set whose Clustered Function score needs to be computed
		
		Returns
		-------
		float
			The Clustered Function evaluation on the given set

		"""

		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, verbose=False):
		"""Find the optimal subset with maximum Clustered Function score for a given budget

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

		if budget >= self.n:
			raise Exception("Budget must be less than groundset size")
		return self.cpp_obj.maximize(optimizer, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose)
	
	def marginalGain(self, X, element):
		"""Find the marginal gain in Clustered Function score when a single item (element) is added to a set (X)

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

		return self.cpp_obj.marginalGain(X, element)
	
	def marginalGainWithMemoization(self, X, element):
		"""Efficiently find the marginal gain in Clustered Function score when a single item (element) is added to a set (X) assuming that memoized statistics for it are already computed

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

		return self.cpp_obj.marginalGainWithMemoization(X, element)

	def evaluateWithMemoization(self, X):
		"""Efficiently compute the Clustered Function score of a set assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			The set whose Clustered Function score needs to be computed
		
		Returns
		-------
		float
			The Clustered Function function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

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
		"""Get the effective ground set of this Clustered Function object. This is equal to the ground set when instantiated with partial=False and is equal to the ground_sub when instantiated with partial=True

		"""
		return self.effective_ground