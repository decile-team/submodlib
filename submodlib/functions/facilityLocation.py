# facilityLocation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from .setFunction import SetFunction
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

	n_jobs: int, optional
		Specifies number of parallel tasks to run while computing sparse matrix. By default, its 1.

	"""

	def __init__(self, n, sijs=None, data=None, mode=None, metric="cosine", num_neigh=-1, partial=False, ground_sub=None, n_jobs=1):
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

		if self.n==0:
			print("ERROR: Number of elements in ground set can't be 0")
			return None

		if self.partial==True and self.ground_sub==None:
			print("ERROR: Ground subset not specified")
			return None
		
		if mode!=None and mode not in ['dense', 'sparse', 'cluster']: # TODO implement code for cluster 
			print("ERROR: Incorrect mode")
			return None

		if type(self.sijs)!=type(None): # User has provided matrix directly: simply consume it
			if type(self.sijs) == scipy.sparse.csr.csr_matrix and num_neigh==-1:
				print("ERROR: num_neigh for given sparse matrix not provided")
				return None
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
			if type(data)!=type(None): # User has only provided data: build similarity matrix and consume it
				
				if self.mode==None:
					self.mode="sparse"

				if self.mode=="dense":
					self.sijs = create_kernel(self.data, self.mode, self.metric, self.num_neigh, n_jobs)
				else:
					self.num_neigh, self.sijs = create_kernel(self.data, self.mode, self.metric, self.num_neigh, n_jobs)
			
			else:
				print("ERROR: Neither data nor similarity matrix provided")
				return None

		
		if self.partial==False: 
			self.cpp_ground_sub = {-1} #Provide a dummy set for pybind11 binding to be successful
		
		#Breaking similarity matrix to simpler native data sturctures for implicit pybind11 binding
		if self.mode=="dense":
			self.cpp_sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure
			self.cpp_obj = FacilityLocation(self.n, self.mode, self.cpp_sijs, self.num_neigh, self.partial, self.cpp_ground_sub)
		
		if self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = FacilityLocation(self.n, self.mode, self.cpp_sijs['arr_val'], self.cpp_sijs['arr_count'], self.cpp_sijs['arr_col'], self.num_neigh, self.partial, self.cpp_ground_sub)
		
		if self.mode=="cluster":
			#TODO
			pass

		
		
		


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
			print("ERROR: X should be a set")
			return None
		
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer):
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

		pass
	
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
			print("ERROR: X should be a set")
			return None
		if type(element)!=int:
			print("ERROR: element should be an int")
			return None

		return self.cpp_obj.marginalGain(X, element)