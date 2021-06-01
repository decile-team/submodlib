# graphCut.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from scipy import sparse
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import GraphCut 
from submodlib.helper import create_kernel, create_cluster_kernels

class GraphCutFunction(SetFunction):
	"""Implementation of the Graph-Cut submodular function.
	
	Graph-Cut models representation and is defined as

	.. math::
			f_{gc}(X) = \\sum_{i \\in V, j \\in X} s_{ij} - \\lambda \\sum_{i, j \\in X} s_{ij}
	
	When :math:`\\lambda` becomes large, graph cut function also tries to model diversity in the subset. :math:`\\lambda` governs the tradeoff between representation and diversity.

	.. note::
			For :math:`\\lambda < 0.5` Graph-Cut is monotone submodular. For :math:`\lambda > 0.5` it is non-monotone submodular.

	Parameters
	----------
	n : int
		Number of elements in the ground set

	lam : float
		Trade-off between representation and diversity as defined by :math:`\\lambda` in the above definition
	
	sijs : list, optional
		Similarity matrix to be used for getting :math:`s_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the similarity matrix

	metric : str, optional
		Similarity metric to be used for computing the similarity matrix
	
	n_neighbors : int, optional
		While constructing similarity matrix, number of nearest neighbors whose similarity values will be kept resulting in a sparse similarity matrix for computation speed up (at the cost of accuracy)
	
	"""

	def __init__(self, n, mode, lambdaVal, separate_master=None, n_master=None, mgsijs=None, ggsijs=None, data=None, data_master=None, num_clusters=None, cluster_labels=None, metric="cosine", num_neighbors=None):
		self.n = n
		self.mode = mode
		self.lambdaVal = lambdaVal
		self.separate_master=separate_master
		self.n_master = n_master
		self.mgsijs = mgsijs
		self.ggsijs = ggsijs
		self.data = data
		self.data_master=data_master
		self.num_clusters=num_clusters
		self.cluster_labels=cluster_labels
		self.metric = metric
		self.num_neighbors = num_neighbors
		
		self.clusters=None
		self.cluster_sijs=None
		self.cluster_map=None
		
		self.cpp_obj = None
		self.cpp_ggsijs = None
		self.cpp_mgsijs = None
		self.cpp_ground_sub = {-1} #Provide a dummy set for pybind11 binding to be successful
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in ['dense', 'sparse']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")
		
		if self.metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if self.separate_master == True:
			if self.n_master is None or self.n_master <=0:
				raise Exception("ERROR: separate master intended but number of elements in master not specified or not positive")	
			if self.mode != "dense":
				raise Exception("Only dense mode supported if separate_master = True")
			if (type(self.mgsijs) != type(None)) and (type(self.mgsijs) != np.ndarray):
				raise Exception("mgsijs provided, but is not dense")
			if (type(self.ggsijs) != type(None)) and (type(self.ggsijs) != np.ndarray):
				raise Exception("ggsijs provided, but is not dense")
			
		if mode == "dense":
			if self.separate_master == True:
				if type(self.mgsijs) == type(None):
					#not provided mgsij - make it
					if (type(data) == type(None)) or (type(data_master) == type(None)):
						raise Exception("Data missing to compute mgsijs")
					if np.shape(self.data)[0]!=self.n or np.shape(self.data_master)[0]!=self.n_master:
						raise Exception("ERROR: Inconsistentcy between n, n_master and no of examples in the given ground data matrix and master data matrix")
					self.mgsijs = np.array(subcp.create_kernel_NS(self.data.tolist(),self.data_master.tolist(), self.metric))
				else:
					#provided mgsijs - verify it's dimensionality
					if np.shape(self.mgsijs)[1]!=self.n or np.shape(self.mgsijs)[0]!=self.n_master:
						raise Exception("ERROR: Inconsistency between n_master, n and no of rows, columns of given mg kernel")

				if type(self.ggsijs) == type(None):
					#not provided ggsijs - make it
					if type(data) == type(None):
						raise Exception("Data missing to compute ggsijs")
					if self.num_neighbors is not None:
						raise Exception("num_neighbors wrongly provided for dense mode")
					self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
					self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
					val = self.cpp_content[0]
					row = list(self.cpp_content[1].astype(int))
					col = list(self.cpp_content[2].astype(int))
					self.ggsijs = np.zeros((n,n))
					self.ggsijs[row,col] = val
				else:
					#provided ggsijs - verify it's dimensionality
					if np.shape(self.ggsijs)[0]!=self.n or np.shape(self.ggsijs)[1]!=self.n:
						raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity gg kernel")

			else:
				if (type(self.ggsijs) == type(None)) and (type(self.mgsijs) == type(None)):
					#no kernel is provided make ggsij kernel
					if type(data) == type(None):
						raise Exception("Data missing to compute ggsijs")
					if self.num_neighbors is not None:
						raise Exception("num_neighbors wrongly provided for dense mode")
					self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
					self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
					val = self.cpp_content[0]
					row = list(self.cpp_content[1].astype(int))
					col = list(self.cpp_content[2].astype(int))
					self.ggsijs = np.zeros((n,n))
					self.ggsijs[row,col] = val
				elif (type(self.ggsijs) == type(None)) and (type(self.mgsijs) != type(None)):
					#gg is not available, mg is - good
					#verify that it is dense and of correct dimension
					if (type(self.mgsijs) != np.ndarray) or np.shape(self.mgsijs)[1]!=self.n or np.shape(self.mgsijs)[0]!=self.n:
						raise Exception("ERROR: Inconsistency between n and no of rows, columns of given kernel")
					self.ggsijs = self.mgsijs
				elif (type(self.ggsijs) != type(None)) and (type(self.mgsijs) == type(None)):
					#gg is available, mg is not - good
					#verify that it is dense and of correct dimension
					if (type(self.ggsijs) != np.ndarray) or np.shape(self.ggsijs)[1]!=self.n or np.shape(self.ggsijs)[0]!=self.n:
						raise Exception("ERROR: Inconsistency between n and no of rows, columns of given kernel")
				else:
					#both are available - something is wrong
					raise Exception("Two kernels have been wrongly provided when separate_master=False")
		elif mode == "sparse":
			if self.separate_master == True:
					raise Exception("Separate master is supported only in dense mode")
			if self.num_neighbors is None or self.num_neighbors <=0:
				raise Exception("Valid num_neighbors is needed for sparse mode")
			if (type(self.ggsijs) == type(None)) and (type(self.mgsijs) == type(None)):
				#no kernel is provided make ggsij sparse kernel
				if type(data) == type(None):
					raise Exception("Data missing to compute ggsijs")
				self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
				val = self.cpp_content[0]
				row = list(self.cpp_content[1].astype(int))
				col = list(self.cpp_content[2].astype(int))
				self.ggsijs = sparse.csr_matrix((val, (row, col)), [n,n])
			elif (type(self.ggsijs) == type(None)) and (type(self.mgsijs) != type(None)):
				#gg is not available, mg is - good
				#verify that it is sparse
				if type(self.mgsijs) != scipy.sparse.csr.csr_matrix:
					raise Exception("Provided kernel is not sparse")
				self.ggsijs = self.mgsijs
			elif (type(self.ggsijs) != type(None)) and (type(self.mgsijs) == type(None)):
				#gg is available, mg is not - good
				#verify that it is dense and of correct dimension
				if type(self.ggsijs) != scipy.sparse.csr.csr_matrix:
					raise Exception("Provided kernel is not sparse")
			else:
				#both are available - something is wrong
				raise Exception("Two kernels have been wrongly provided when separate_master=False")

		if self.separate_master==None:
			self.separate_master = False

		if self.mode=="dense" and self.separate_master == False :
			self.cpp_ggsijs = self.ggsijs.tolist() #break numpy ndarray to native list of list datastructure
			
			if type(self.cpp_ggsijs[0])==int or type(self.cpp_ggsijs[0])==float: #Its critical that we pass a list of list to pybind11
																			 #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(self.cpp_ggsijs)
				self.cpp_ggsijs=l

			self.cpp_obj = GraphCut(self.n, self.cpp_ggsijs, False, self.cpp_ground_sub, self.lambdaVal)
		
		elif self.mode=="dense" and self.separate_master == True :
			self.cpp_ggsijs = self.ggsijs.tolist() #break numpy ndarray to native list of list datastructure
			
			if type(self.cpp_ggsijs[0])==int or type(self.cpp_ggsijs[0])==float: #Its critical that we pass a list of list to pybind11
																			 #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(self.cpp_ggsijs)
				self.cpp_ggsijs=l
			
			self.cpp_mgsijs = self.mgsijs.tolist() #break numpy ndarray to native list of list datastructure
			
			if type(self.cpp_mgsijs[0])==int or type(self.cpp_mgsijs[0])==float: #Its critical that we pass a list of list to pybind11
																			 #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(self.cpp_mgsijs)
				self.cpp_mgsijs=l

			self.cpp_obj = GraphCut(self.n, self.cpp_mgsijs, self.cpp_ggsijs, self.lambdaVal)

		elif self.mode == "sparse":
			self.cpp_ggsijs = {}
			self.cpp_ggsijs['arr_val'] = self.ggsijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_ggsijs['arr_count'] = self.ggsijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_ggsijs['arr_col'] = self.ggsijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = GraphCut(self.n, self.cpp_ggsijs['arr_val'], self.cpp_ggsijs['arr_count'], self.cpp_ggsijs['arr_col'], lambdaVal)
		else:
			raise Exception("Invalid")

		self.effective_ground = self.cpp_obj.getEffectiveGroundSet()

	