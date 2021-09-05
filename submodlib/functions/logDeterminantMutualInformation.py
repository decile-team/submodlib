# logDeterminantMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import LogDeterminantMutualInformation 
from submodlib.helper import create_kernel

class LogDeterminantMutualInformationFunction(SetFunction):
	"""Implementation of the Log Determinant Mutual Information (LogDetMI) function.

	Given a :ref:`functions.submodular-mutual-information` function, Log Determinant Mutual Information function is its instantiation using a :class:`~submodlib.functions.logDeterminant.LogDeterminantFunction`. 
	
	Let :math:`S_{A, B}` be the cross-similarity matrix between the items in sets :math:`A` and :math:`B`. Also, denote :math:`S_{AB} = S_{A \\cup B}`.

	We construct a similarity matrix :math:`S^{\\eta}` (on a base matrix :math:`S`) in such a way that the cross-similarity between :math:`A` and :math:`Q` is multiplied by :math:`\\eta` (i.e :math:`S^{\\eta}_{A,Q} = \\eta S_{A,Q}`) to control the trade-off between query relevance and diversity. Higher values of :math:`\\eta` ensure greater query-relevance while lower values favor diversity. 
	
	Using a similarity matrix defined above and with :math:`f(A) = \\log\\det(S^{\\eta}_{A})`, we have: 
	
	.. math::
			I_f(A; Q) = \\log\\det(S_{A}) -\\log\\det(S_{A} - \\eta^2 S_{A,Q}S_{Q}^{-1}S_{A,Q}^T)
			
	.. note::
			The query-DPP considered in :cite:`sharghi2016query,sharghi2017query` is a special case of LogDetMI.
	
	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	num_queries : int
		Number of query points in the target.
	
	lambdaVal : float
		Addition to :math:`s_{ii} (1)` so that :math:`\\log` doesn't become 0
	
	data_sijs : numpy.ndarray, optional
		Similarity kernel between the elements of the ground set. Shape: n X n. When not provided, it is computed using data.
	
	query_sijs : numpy.ndarray, optional
		Similarity kernel between the ground set and the queries. Shape: n X num_queries. When not provided, it is computed using data, queryData and metric.
	
	query_query_sijs : numpy.ndarray, optional
		Similarity kernel between the query points. Shape: num_queries X num_queries. When not provided, it is computed using queryData.
	
	data : numpy.ndarray, optional
		Matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. Mandatory if either of data_sijs or query_sijs is not provided. Ignored if both data_sijs and query_sijs are provided.
	
	queryData : numpy.ndarray, optional
		Matrix of shape num_queries X num_features containing the query elements. queryData[i] should contain the num-features dimensional features of query i. It is optional (and is ignored if provided) if query_sijs has been provided.

	metric : str, optional
		Similarity metric to be used for computing the similarity kernels. Can be "cosine" for cosine similarity or "euclidean" for similarity based on euclidean distance. Default is "cosine". 
	
	magnificationEta : float, optional
		The value of the query-relevance vs diversity trade-off. Increasing :math:`\eta` tends to increase query-relevance while reducing query-coverage and diversity. Default is 1.

	"""

	def __init__(self, n, num_queries, lambdaVal, data_sijs=None, query_sijs=None, query_query_sijs=None, data=None, queryData=None, metric="cosine", magnificationEta=1):
		self.n = n
		self.num_queries = num_queries
		self.lambdaVal=lambdaVal
		self.metric = metric
		self.magnificationEta=magnificationEta
		self.data_sijs = data_sijs
		self.query_sijs = query_sijs
		self.query_query_sijs = query_query_sijs
		self.data = data
		self.queryData = queryData
		
		self.cpp_obj = None
		self.cpp_data_sijs = None
		self.cpp_query_sijs = None
		self.cpp_query_query_sijs = None
		self.cpp_content = None
		self.cpp_content2 = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.num_queries < 0:
			raise Exception("ERROR: Number of queries must be >= 0")

		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if (type(self.data_sijs) != type(None)) and (type(self.query_sijs) != type(None)) and (type(self.query_query_sijs) != type(None)): # User has provided all three kernels
			if type(self.data_sijs) != np.ndarray:
				raise Exception("Invalid data kernel type provided, must be ndarray")
			if type(self.query_sijs) != np.ndarray:
				raise Exception("Invalid query kernel type provided, must be ndarray")
			if type(self.query_query_sijs) != np.ndarray:
				raise Exception("Invalid query-query kernel type provided, must be ndarray")
			if np.shape(self.data_sijs)[0]!=self.n or np.shape(self.data_sijs)[1]!=self.n:
				raise Exception("ERROR: data kernel should be n X n")
			if np.shape(self.query_sijs)[0]!=self.n or np.shape(self.query_sijs)[1]!=self.num_queries:
				raise Exception("ERROR: Query Kernel should be n X num_queries")
			if np.shape(self.query_query_sijs)[0]!=self.num_queries or np.shape(self.query_query_sijs)[1]!=self.num_queries:
				raise Exception("ERROR: Query-query Kernel should be num_queries X num_queries")
			if (type(self.data) != type(None)) or (type(self.queryData) != type(None)):
				print("WARNING: similarity kernels found. Provided data and query matrices will be ignored.")
		else: #similarity kernels have not been provided
			if (type(self.data) == type(None)) or (type(self.queryData) == type(None)):
				raise Exception("Since kernels are not provided, data matrices are a must")
			if np.shape(self.data)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
			if np.shape(self.queryData)[0]!=self.num_queries:
				raise Exception("ERROR: Inconsistentcy between num_queries and no of examples in the given query data matrix")
			
			#construct imageKernel
			self.num_neighbors = self.n #Using all data as num_neighbors in case of dense mode
			self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
			val = self.cpp_content[0]
			row = list(self.cpp_content[1].astype(int))
			col = list(self.cpp_content[2].astype(int))
			self.data_sijs = np.zeros((self.n,self.n))
			self.data_sijs[row,col] = val
		
		    #construct queryKernel
			self.query_sijs = np.array(subcp.create_kernel_NS(self.queryData.tolist(),self.data.tolist(), self.metric))

			#construct queryQueryKernel
			self.num_neighbors2 = self.num_queries #Using all data as num_neighbors in case of dense mode
			self.cpp_content2 = np.array(subcp.create_kernel(self.queryData.tolist(), self.metric, self.num_neighbors2))
			val2 = self.cpp_content2[0]
			row2 = list(self.cpp_content2[1].astype(int))
			col2 = list(self.cpp_content2[2].astype(int))
			self.query_query_sijs = np.zeros((self.num_queries,self.num_queries))
			self.query_query_sijs[row2,col2] = val2
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		self.cpp_data_sijs = self.data_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_data_sijs[0])==int or type(self.cpp_data_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_data_sijs)
			self.cpp_data_sijs=l
		
		self.cpp_query_sijs = self.query_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_query_sijs[0])==int or type(self.cpp_query_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_query_sijs)
			self.cpp_query_sijs=l
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		self.cpp_query_query_sijs = self.query_query_sijs.tolist() #break numpy ndarray to native list of list datastructure
		
		if type(self.cpp_query_query_sijs[0])==int or type(self.cpp_query_query_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
			l=[]
			l.append(self.cpp_query_query_sijs)
			self.cpp_query_query_sijs=l

		self.cpp_obj = LogDeterminantMutualInformation(self.n, self.num_queries, self.cpp_data_sijs, self.cpp_query_sijs, self.cpp_query_query_sijs, self.lambdaVal, self.magnificationEta)
		self.effective_ground = set(range(n))

	