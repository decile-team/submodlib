# disparitySum.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from scipy import sparse
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import DisparitySum 
from submodlib.helper import create_kernel, create_cluster

class DisparitySumFunction(SetFunction):
	"""Implementation of the Disparity-Sum function.

	Disparity-Sum models diversity by computing the sum of pairwise distances of all the elements in a subset. It is defined as

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

	def __init__(self, n, sijs=None, data=None, mode=None, metric="cosine", num_neigh=-1, partial=False, ground_sub=None):
		self.n = n
		self.mode = mode
		self.metric = metric
		self.sijs = sijs
		self.data = data
		self.num_neigh = num_neigh
		self.partial = partial
		self.ground_sub = ground_sub
		self.cpp_obj = None
		self.cpp_sijs = None
		self.cpp_ground_sub = ground_sub
		self.cpp_content = None

		if self.n==0:
			raise Exception("ERROR: Number of elements in ground set can't be 0")

		if self.partial==True and self.ground_sub==None:
			raise Exception("ERROR: Ground subset not specified")
		
		if mode!=None and mode not in ['dense', 'sparse']:
			raise Exception("ERROR: Incorrect mode")
		
		if metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric")

		if type(self.sijs)!=type(None): # User has provided sim matrix directly: simply consume it
			if np.shape(self.sijs)[0]!=self.n:
				raise Exception("ERROR: Inconsistentcy between n and no of examples in the given similarity matrix")
			
			if type(self.sijs) == scipy.sparse.csr.csr_matrix and num_neigh==-1:
				raise Exception("ERROR: num_neigh for given sparse matrix not provided")
			if self.mode!=None: # Ensure that there is no inconsistency in similarity matrix and provided mode
				if type(self.sijs) == np.ndarray and self.mode!="dense":
					print("WARNING: Incorrect mode provided for given similarity matrix, changing it to dense")
					self.mode="dense"
				if type(self.sijs) == scipy.sparse.csr.csr_matrix and self.mode!="sparse":
					print("WARNING: Incorrect mode provided for given similarity matrix, changing it to sparse")
					self.mode="sparse"
			else: # Infer mode from similarity matrix
				if type(self.sijs) == np.ndarray:
					self.mode="dense"
				if type(self.sijs) == scipy.sparse.csr.csr_matrix:
					self.mode="sparse"
		else:
			if type(self.data)!=type(None): # User has only provided data: build similarity matrix/cluster-info and consume it
				
				if np.shape(self.data)[0]!=self.n:
					raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")

				if self.mode==None:
					self.mode="sparse"

				if self.num_neigh==-1:
					self.num_neigh=np.shape(self.data)[0] #default is total no of datapoints

				
				self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neigh))
				val = self.cpp_content[0]
				row = list(map(lambda arg: int(arg), self.cpp_content[1]))
				col = list(map(lambda arg: int(arg), self.cpp_content[2]))
				if self.mode=="dense":
					self.sijs = np.zeros((n,n))
					self.sijs[row,col] = val
				if self.mode=="sparse":
					self.sijs = sparse.csr_matrix((val, (row, col)), [n,n])

			else:
				raise Exception("ERROR: Neither data nor similarity matrix provided")
		
		if self.partial==False: 
			self.cpp_ground_sub = {-1} #Provide a dummy set for pybind11 binding to be successful
		
		#Breaking similarity matrix to simpler native data sturctures for implicit pybind11 binding
		if self.mode=="dense":
			self.cpp_sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure
			
			if type(self.cpp_sijs[0])==int or type(self.cpp_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																			 #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
				l=[]
				l.append(self.cpp_sijs)
				self.cpp_sijs=l
			if np.shape(self.cpp_sijs)[0]!=np.shape(self.cpp_sijs)[1]: #TODO: relocate this check to some earlier part of code
				raise Exception("ERROR: Dense similarity matrix should be a square matrix")

			self.cpp_obj = DisparitySum(self.n, self.mode, self.cpp_sijs, self.num_neigh, self.partial, self.cpp_ground_sub)
		
		if self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = DisparitySum(self.n, self.mode, self.cpp_sijs['arr_val'], self.cpp_sijs['arr_count'], self.cpp_sijs['arr_col'], self.num_neigh, self.partial, self.cpp_ground_sub)
		
		self.cpp_ground_sub=self.cpp_obj.getEffectiveGroundSet()
		self.ground_sub=self.cpp_ground_sub

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
		budget : int
			Desired size of the optimal set
		optimizer : optimizers.Optimizer
			The optimizer that should be used to compute the optimal set

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


	def marginalGainSequential(self, X, element):
		return self.cpp_obj.marginalGainSequential(X, element)

	def evaluateSequential(self, X):
		return self.cpp_obj.evaluateSequential(X)

	def sequentialUpdate(self, X, element):
		self.cpp_obj.sequentialUpdate(X, element)
	
	def clearPreCompute(self):
		self.cpp_obj.clearPreCompute()
	
	def getEffectiveGroundSet(self):
		return self.cpp_obj.getEffectiveGroundSet()