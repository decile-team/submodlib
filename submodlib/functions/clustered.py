# clustered.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
from .setFunction import SetFunction
from submodlib.helper import create_cluster
import submodlib_cpp as subcp

class ClusteredFunction(SetFunction):
	"""Implementation of the Clustered function.

	Given a function and a clustering, clustered function internally creates a mixture of function on each cluster. It is defined as
	
	.. math::
			f(X) = \\sum_i f_{C_i}(X \\cap C_i)
	
	.. note::
			When the clusters are labels, this becomes supervised subset selection.

	Parameters
	----------
	n : int
		Number of elements in the ground set
	f_name : string
		Name of particular instantiated set function whose clustered version is desired
	data : numpy ndarray, optional
		Data matrix which will be used for computing the similarity matrix
	cluster_lab : list, optional
		Its a list that contains cluster label corrosponding to ith datapoint
	metric : string
		similarity metric to be used while computing similarity kernel for each cluster. By default, its cosine
	num_cluster : int, optional
		number of clusters to be created (if only data matrix is provided) or number of clusters being used (if precreated cluster labels are also provided along with data matrix).
		Note that num_cluster must be provided if cluster_lab has been provided 
		
	"""

	def __init__(self, n, f_name, data, cluster_lab=None, metric="cosine", num_cluster=None):
		self.n = n
		self.f_name = f_name
		self.metric = metric
		self.data = data
		self.clusters=None
		self.cluster_sijs=None
		self.cluster_map=None
		self.cluster_lab=cluster_lab
		self.num_cluster=num_cluster
		self.cpp_ground_sub=None
		
		if self.n==0:
			raise Exception("ERROR: Number of elements in ground set can't be 0")

		#print(metric)
		if metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric")

		self.clusters, self.cluster_sijs, self.cluster_map = create_cluster(self.data.tolist(), self.metric, self.cluster_lab, self.num_cluster)
		l_temp = []
		for el in self.cluster_sijs:
			temp=el.tolist()
			if type(temp[0])==int or type(temp[0])==float: #Its critical that we pass a list of list to pybind11
															#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(temp)
				temp=l
			l_temp.append(temp)
		self.cluster_sijs = l_temp
		self.cpp_obj = subcp.ClusteredFunction(self.n, self.f_name, self.clusters, self.cluster_sijs, self.cluster_map)
		self.cpp_ground_sub=self.cpp_obj.getEffectiveGroundSet()

	def evaluate(self, X):
		"""Computes the score of a set

		Parameters
		----------
		X : set
			The set whose score needs to be computed
		
		Returns
		-------
		float
			The function evaluation on the given set

		"""

		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.cpp_ground_sub)==False:
			raise Exception("ERROR: X is not a subset of ground set")
		
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbosity=False):
		"""Find the optimal subset with maximum score

		Parameters
		----------
		optimizer : optimizers.Optimizer
			The optimizer that should be used to compute the optimal set
		
		budget : int
			Desired size of the optimal set
		
		stopIfZeroGain : bool


		stopIfNegativeGain : bool


		verbosity : bool

		Returns
		-------
		set
			The optimal set of size budget

		"""

		return self.cpp_obj.maximize(optimizer, budget, stopIfZeroGain, stopIfNegativeGain, verbosity)
	
	def marginalGain(self, X, element):
		"""Find the marginal gain of adding an item to a set

		Parameters
		----------
		X : set
			Set on which the marginal gain of adding an element has to be calculated
		element : int
			Element for which the marginal gain is to be calculated

		Returns
		-------
		float
			Marginal gain of adding element to X

		"""

		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if type(element)!=int:
			raise Exception("ERROR: element should be an int")

		if X.issubset(self.cpp_ground_sub)==False:
			raise Exception("ERROR: X is not a subset of ground set")

		return self.cpp_obj.marginalGain(X, element)