# clustered.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
from .setFunction import SetFunction
from submodlib.helper import create_cluster_kernels
import submodlib_cpp as subcp
from submodlib_cpp import Clustered 

class ClusteredFunction(SetFunction):
	"""Implementation of the Clustered function.

	Given a set-function :math:`f` and a clustering, clustered function internally creates a mixture of functions each defined over a cluster. It is thus defined as
	
	.. math::
			f(X) = \\sum_i f_{C_i}(X)

	where :math:`f_{C_i}` operates only on cluster :math:`C_i` as sub-groundset and interprets :math:`X` as :math:`X \\cap C_i`.
	
	.. note::
			When the clusters are labels, ClusteredFunction is useful to achieve supervised subset selection.
	
	.. note::
			Some functions in this toolkit provide a "clustered" mode operation, achieving the same effect as invoking ClusteredFunction on those functions.

	Parameters
	----------
	n : int
		Number of elements in the ground set. Must be > 0.
	f_name : str
		Name of particular set function whose clustered implementation is desired.
	data : numpy.ndarray
		Data matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. This is used for computing the similarity kernel (and for computing the clusters if clustering is not provided). 
	mode : str
		Governs the internal implementation details. Can be "single" (to create a single dense large similarity kernel) or "multi" (to create one small dense kernel per cluster). If "single", internally the "partial" versions of the functions are used to get the functions for each cluster. If "multi", the functions for each cluster are instantiated separately with each cluster corresponding to a different groundset. 
	cluster_lab : list, optional
		List of size n, containing the cluster labels for each data point. If not provided, clustering is done internally using sklearn's `BIRCH <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html>`_ using provided data matrix.
	num_clusters : int, optional
		Number of clusters. Mandatory if cluster_lab is provided. If cluster_lab is not provided, clustering is done internally. In this case if num_clusters is not provided, an optimal number of clusters is created based on the supplied data. 
	metric : str, optional
		Similarity metric to be used while computing similarity kernel for each cluster (in "multi" mode) or a single dense kernel (in "single" mode). Can be "euclidean" or "cosine". Default value is "cosine".
	lambdaVal : float, optional
		Additional parameter that needs to be passed on to the set function if required. For example, the additional parameter of :class:`~submodlib.functions.graphCut.GraphCutFunction` and :class:`~submodlib.functions.logDeterminant.LogDeterminantFunction`. Default is 1.
		
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
		self.lambdaVal = lambdaVal

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in ['single', 'multi']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'single' or 'multi'")

		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric")

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
			# print("self.n: ", self.n)
			# print("self.f_name: ", self.f_name)
			# print("self.clusters: ", self.clusters)
			# print("self.cpp_sijs: ", self.cpp_sijs)
			# print("self.lambdaVal: ", self.lambdaVal)
			self.cpp_obj = Clustered(self.n, self.f_name, self.clusters, self.cpp_sijs, self.lambdaVal)
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
			# print("self.n: ", self.n)
			# print("self.f_name: ", self.f_name)
			# print("self.clusters: ", self.clusters)
			# print("self.cluster_sijs: ", self.cluster_sijs)
			# print("self.cluster_map: ", self.cluster_map)
			# print("self.lambdaVal: ", self.lambdaVal)
			self.cpp_obj = Clustered(self.n, self.f_name, self.clusters, self.cluster_sijs, self.cluster_map, lambdaVal)
		self.effective_ground=self.cpp_obj.getEffectiveGroundSet()