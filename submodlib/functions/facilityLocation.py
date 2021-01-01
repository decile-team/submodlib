# facilityLocation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from scipy import sparse
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FacilityLocation 
from submodlib.helper import create_kernel

class FacilityLocationFunction(SetFunction):
	"""Implementation of the Facility-Location submodular function.
	
	Facility-Location function :cite:`mirchandani1990discrete` attempts to model representation, as in it tries to find a representative subset of items, akin to centroids and medoids in clustering. The Facility-Location function is closely related to k-medoid clustering. While diversity *only* looks at the elements in the chosen subset, representativeness also worries about their similarity with the remaining elements in the superset. Denote :math:`s_{ij}` as the similarity between images/datapoints :math:`i` and :math:`j`. We can then define 

	.. math::
			f(X) = \\sum_{i \\in V} \\max_{j \\in X} s_{ij} 
	
	For each image :math:`i` in the ground set :math:`V`, we compute the representative from subset :math:`X` which is closest to :math:`i` and add these similarities for all images. 

	Facility-Location is monotone submodular.
	
	.. note:: 
		This function requires computing a :math:`\\mathcal{O}(n^2)` similarity function. However, as shown in :cite:`wei2014fast`, we can approximate this with a nearest neighbor graph, which will require much less storage, and also can run much faster for large ground set sizes.

	Parameters
	----------

	n : int
		Number of elements in the ground set
	
	sijs : numpy ndarray or scipy sparse matrix, optional
		Similarity matrix to be used for getting :math:`s_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : numpy ndarray, optional
		Data matrix which will be used for computing the similarity matrix

	mode: str, optional
		It specifies weather similarity matrix will be dense or sparse. By default, its sparse

	metric : str, optional
		Similarity metric to be used for computing the similarity matrix. By default, its cosine
	
	num_neigh : int, optional
		While constructing similarity matrix, number of nearest neighbors whose similarity values will be kept resulting in a sparse similarity matrix for computation speed up (at the cost of accuracy)

	partial: bool, optional
		if True, a subset of ground set will be used. By default, its False. 

	ground_sub: set, optional
		Specifies subset of ground set that will be used when partial is True. 

	"""

	def __init__(self, n, n_master=-1, sijs=None, data=None, data_master=None, mode=None, metric="cosine", num_neigh=-1, partial=False, ground_sub=None):
		self.n = n
		self.n_master = n_master
		self.mode = mode
		self.metric = metric
		self.sijs = sijs
		self.data = data
		self.data_master=data_master
		self.num_neigh = num_neigh
		self.partial = partial
		self.ground_sub = ground_sub
		self.seperateMaster=False
		self.cpp_obj = None
		self.cpp_sijs = None
		self.cpp_ground_sub = ground_sub
		self.cpp_content = None

		if self.n==0:
			raise Exception("ERROR: Number of elements in ground set can't be 0")

		if self.partial==True and self.ground_sub==None:
			raise Exception("ERROR: Ground subset not specified")
		
		if mode!=None and mode not in ['dense', 'sparse', 'cluster']: # TODO implement code for cluster 
			raise Exception("ERROR: Incorrect mode")
		
		if metric not in ['euclidean', 'cosine']:
			raise Exception("ERROR: Unsupported metric")

		if type(self.sijs)!=type(None): # User has provided matrix directly: simply consume it
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
			if type(self.data)!=type(None): # User has only provided data: build similarity matrix and consume it
				
				if np.shape(self.data)[0]!=self.n:
					raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")

				if type(self.data_master)!=type(None):
					self.seperateMaster=True
					if np.shape(self.data_master)[0]!=self.n_master:
						raise Exception("ERROR: Inconsistentcy between n_master and no of examples in the given data_master matrix")
					if self.mode=="sparse" or self.mode=="cluster":
						raise Exception("ERROR: mode can't be sparse or cluster if ground and master datasets are different")
					if partial==True:
						raise Exception("ERROR: partial can't be True if ground and master datasets are different")

				if self.mode==None:
					self.mode="sparse"

				if self.num_neigh==-1 and self.seperateMaster==False:
					self.num_neigh=np.shape(self.data)[0] #default is total no of datapoints

				if self.seperateMaster==True: #mode in this case will always be dense
					self.sijs = np.array(subcp.create_kernel_NS(self.data.tolist(),self.data_master.tolist(), self.metric))
				else:
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
			if np.shape(self.cpp_sijs)[0]!=np.shape(self.cpp_sijs)[1] and self.seperateMaster==False: #TODO: relocate this check to some earlier part of code
				raise Exception("ERROR: Dense similarity matrix should be a square matrix if ground and master datasets are same")

			self.cpp_obj = FacilityLocation(self.n, self.mode, self.cpp_sijs, self.num_neigh, self.partial, self.cpp_ground_sub, self.seperateMaster)
		
		if self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = FacilityLocation(self.n, self.mode, self.cpp_sijs['arr_val'], self.cpp_sijs['arr_count'], self.cpp_sijs['arr_col'], self.num_neigh, self.partial, self.cpp_ground_sub)
		
		if self.mode=="cluster":
			#TODO
			pass

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