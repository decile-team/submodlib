# logDeterminantConditionalMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import LogDeterminantConditionalMutualInformation 
from submodlib.helper import create_kernel

class LogDeterminantConditionalMutualInformationFunction(SetFunction):
	"""Implementation of the Log Determinant Conditional Mutual Information (LogDetCMI) function.

	Given a :ref:`functions.conditional-mutual-information` function, Log Determinant Conditional Mutual Information function is its instantiation using a :class:`~submodlib.functions.logDeterminant.LogDeterminantFunction`. 

	Let :math:`S_{A, B}` be the cross-similarity matrix between the items in sets :math:`A` and :math:`B`. Also, denote :math:`S_{AB} = S_{A \\cup B}`.

	We construct a similarity matrix :math:`S^{\\eta,\\nu}` (on a base matrix :math:`S`) in such a way that the cross-similarity between :math:`A` and :math:`Q` is multiplied by :math:`\\eta` (i.e :math:`S^{\\eta}_{A,Q} = \\eta S_{A,Q}`) to control the query-relevance and diversity trade-off and between :math:`A` and :math:`P` is multiplied by :math:`\\nu` (i.e :math:`S^{\\nu}_{A,P} = \\nu S_{A,P}`) to control the hardness of enforcing privacy constraints. 
	
	Using a similarity matrix defined above and with :math:`f(A) = \\log\\det(S^{\\nu}_{A})`, we have: 
	
	.. math::
			I_f(A; Q|P ) = \\log \\frac{\\det(I - S_{P}^{-1} S_{P, Q} S_{Q}^{-1} S_{P, Q}^T)}{\\det(I - S_{A P}^{-1} S_{A P, Q} S_{Q}^{-1} S_{A P, Q}^T)}
	
	.. note::
			LogDetCMI favors query-relevance and privacy-irrelevance over query-coverage and diversity.
	
	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	num_queries : int
		Number of query points in the target.
	
	num_privates : int
		Number of private instances in the target.
	
	lambdaVal : float
		Addition to :math:`s_{ii} (1)` so that :math:`\\log` doesn't become 0
	
	data_sijs : numpy.ndarray, optional
		Similarity kernel between the elements of the ground set. Shape: n X n. When not provided, it is computed using data.
	
	query_sijs : numpy.ndarray, optional
		Similarity kernel between the ground set and the queries. Shape: n X num_queries. When not provided, it is computed using data, queryData and metric.
	
	query_query_sijs : numpy.ndarray, optional
		Similarity kernel between the query points. Shape: num_queries X num_queries. When not provided, it is computed using queryData.
	
	private_sijs : numpy.ndarray, optional
		Similarity kernel between the ground set and the private instances. Shape: n X num_privates. When not provided, it is computed using data and privateData.
	
	private_private_sijs : numpy.ndarray, optional
		Similarity kernel between the private instances. Shape: num_privates X num_privates. When not provided, it is computed using privateData.
	
	query_private_sijs : numpy.ndarray, optional
		Similarity kernel between the query instances and the private instances. Shape: num_queries X num_privates. When not provided, it is computed using queryData and privateData.

	data : numpy.ndarray, optional
		Matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. Mandatory, if either if data_sijs or private_sijs is not provided. Ignored if both data_sijs and private_sijs are provided.
	
	queryData : numpy.ndarray, optional
		Matrix of shape num_queries X num_features containing the query elements. queryData[i] should contain the num-features dimensional features of query i. It is optional (and is ignored if provided) if query_sijs has been provided.

	privateData : numpy.ndarray, optional
		Matrix of shape num_privates X num_features containing the private instances. privateData[i] should contain the num-features dimensional features of private instance i. Must be provided if private_sijs is not provided. Ignored if both data_sijs and private_sijs are provided.

	metric : str, optional
		Similarity metric to be used for computing the similarity kernels. Can be "cosine" for cosine similarity or "euclidean" for similarity based on euclidean distance. Default is "cosine". 
	
	magnificationEta : float, optional
		The value of the query-relevance vs diversity trade-off. Increasing :math:`\eta` tends to increase query-relevance while reducing query-coverage and diversity. Default is 1.

	privacyHardness : float, optional
		Parameter that governs the hardness of the privacy constraint. Default is 1.
	
	"""

	def __init__(self, n, num_queries, num_privates, lambdaVal, data_sijs=None, query_sijs=None, query_query_sijs=None, private_sijs=None, private_private_sijs=None, query_private_sijs=None, data=None, queryData=None, privateData=None, metric="cosine", magnificationEta=1, privacyHardness=1):
		self.n = n
		self.num_queries = num_queries
		self.num_privates = num_privates
		self.lambdaVal=lambdaVal
		self.metric = metric
		self.magnificationEta=magnificationEta
		self.privacyHardness=privacyHardness
		self.data_sijs = data_sijs
		self.query_sijs = query_sijs
		self.query_query_sijs = query_query_sijs
		self.private_sijs = private_sijs
		self.private_private_sijs = private_private_sijs
		self.query_private_sijs = query_private_sijs
		self.data = data
		self.queryData = queryData
		self.privateData = privateData
		
		self.cpp_obj = None
		self.cpp_data_sijs = None
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

		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if (type(self.data_sijs) != type(None)) and (type(self.query_sijs) != type(None)) and (type(self.query_query_sijs) != type(None)) and (type(self.private_sijs) != type(None)) and (type(self.private_private_sijs) != type(None)) and (type(self.query_private_sijs) != type(None)): # User has provided all required kernels
			if type(self.data_sijs) != np.ndarray:
				raise Exception("Invalid data kernel type provided, must be ndarray")
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
			if np.shape(self.data_sijs)[0]!=self.n or np.shape(self.data_sijs)[1]!=self.n:
				raise Exception("ERROR: data kernel should be n X n")
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
			if (type(self.data) != type(None)) or (type(self.queryData) != type(None)) or (type(self.privateData) != type(None)):
				print("WARNING: similarity kernels found. Provided data, query and private matrices will be ignored.")
		else: #similarity kernels have not been provided
			if (type(self.data) == type(None)) or (type(self.queryData) == type(None)) or (type(self.privateData) == type(None)):
				raise Exception("Since kernels are not provided, data matrices are a must")
			if np.shape(self.data)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
			if np.shape(self.queryData)[0]!=self.num_queries:
				raise Exception("ERROR: Inconsistentcy between num_queries and no of examples in the given query data matrix")
			if np.shape(self.privateData)[0]!=self.num_privates:
				raise Exception("ERROR: Inconsistentcy between num_privates and no of examples in the given private data matrix")
			
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

			#construct privateKernel
			self.private_sijs = np.array(subcp.create_kernel_NS(self.privateData.tolist(),self.data.tolist(), self.metric))

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

		self.cpp_obj = LogDeterminantConditionalMutualInformation(self.n, self.num_queries, self.num_privates, self.cpp_data_sijs, self.cpp_query_sijs, self.cpp_query_query_sijs, self.cpp_private_sijs, self.cpp_private_private_sijs,self.cpp_query_private_sijs, self.lambdaVal, self.magnificationEta, self.privacyHardness)
		self.effective_ground = set(range(n))

	