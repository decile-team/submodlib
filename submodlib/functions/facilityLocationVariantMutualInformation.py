# facilityLocationVariantMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FacilityLocationVariantMutualInformation 
from submodlib.helper import create_kernel

class FacilityLocationVariantMutualInformationFunction(SetFunction):
	"""Implementation of the Facility Location Variant Mutual Information (FL2MI) function.

	Given a :ref:`functions.submodular-mutual-information` function, Facility Location Variant Mutual Information function is its instantiation using a :class:`~submodlib.functions.facilityLocation.FacilityLocationFunction`. However it is slightly different from :class:`~submodlib.functions.facilityLocationMutualInformation.FacilityLocationMutualInformationFunction`. This variant considers only cross-similarities between data points and the target. Mathematically, it takes the following form:

	.. math::
			I_f(A; Q) = \\sum_{i \\in Q} \\max_{j \\in A} s_{ij} + \\eta \\sum_{i \\in A} \\max_{j \\in Q} s_{ij}
	
	This expression has interesting characteristics different from those of FL1MI. In particular, there is no saturation in FL2MI and it just models the pairwise similarities of target to data points and vice versa.
	
	.. note::
			CRAIG :cite:`mirzasoleiman2020coresets` when applied to the task of targeted subset selection can be seen as a special case of FL2MI (see :cite:`kaushal2021prism`).

	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	num_queries : int
		Number of query points in the target.
	
	query_sijs : numpy.ndarray, optional
		Similarity kernel between the ground set and the queries. Shape: n X num_queries. When not provided, it is computed using data, queryData and metric.
	
	data : numpy.ndarray, optional
		Matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. Mandatory, if query_sijs is not provided. Ignored if query_sijs is provided.
	
	queryData : numpy.ndarray, optional
		Matrix of shape num_queries X num_features containing the query elements. queryData[i] should contain the num-features dimensional features of query i. It is optional (and is ignored if provided) if query_sijs has been provided.

	metric : str, optional
		Similarity metric to be used for computing the similarity kernels. Can be "cosine" for cosine similarity or "euclidean" for similarity based on euclidean distance. Default is "cosine". 
	
	queryDiversityEta : float, optional
		The value of the query-relevance vs diversity trade-off. Increasing :math:`\eta` tends to increase query-relevance while reducing query-coverage and diversity. Default is 1.

	"""

	def __init__(self, n, num_queries, query_sijs=None, data=None, queryData=None, metric="cosine", queryDiversityEta=1):
		self.n = n
		self.num_queries = num_queries
		self.metric = metric
		self.query_sijs = query_sijs
		self.data = data
		self.queryData = queryData
		self.queryDiversityEta=queryDiversityEta
		self.cpp_obj = None
		self.cpp_query_sijs = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.num_queries < 0:
			raise Exception("ERROR: Number of queries must be >= 0")

		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if type(self.query_sijs) != type(None): # User has provided query kernel
			if type(self.query_sijs) != np.ndarray:
				raise Exception("Invalid query kernel type provided, must be ndarray")
			if np.shape(self.query_sijs)[0]!=self.n or np.shape(self.query_sijs)[1]!=self.num_queries:
				raise Exception("ERROR: Query Kernel should be n X num_queries")
			if (type(self.data) != type(None)) or (type(self.queryData) != type(None)):
				print("WARNING: similarity query kernel found. Provided data and query matrices will be ignored.")
		else: #similarity query kernel has not been provided
			if (type(self.data) == type(None)) or (type(self.queryData) == type(None)):
				raise Exception("Since query kernel is not provided, data matrices are a must")
			if np.shape(self.data)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
			if np.shape(self.queryData)[0]!=self.num_queries:
				raise Exception("ERROR: Inconsistentcy between num_queries and no of examples in the given query data matrix")
			
		    #construct queryKernel
			self.query_sijs = np.array(subcp.create_kernel_NS(self.queryData.tolist(),self.data.tolist(), self.metric))
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		self.cpp_query_sijs = self.query_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_query_sijs[0])==int or type(self.cpp_query_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_query_sijs)
			self.cpp_query_sijs=l

		self.cpp_obj = FacilityLocationVariantMutualInformation(self.n, self.num_queries, self.cpp_query_sijs, self.queryDiversityEta)
		self.effective_ground = set(range(n))

	