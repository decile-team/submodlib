# facilityLocation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
import scipy
from scipy import sparse
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FacilityLocation
from submodlib_cpp import FacilityLocation2 
from submodlib.helper import create_kernel, create_cluster_kernels
#from memory_profiler import profile

class FacilityLocationFunction(SetFunction):
	"""Implementation of the Facility Location submodular function (FL).

	Facility-Location function :cite:`mirchandani1990discrete` attempts to model representation, as in it tries to find a representative subset of items, akin to centroids and medoids in clustering. The Facility-Location function is closely related to k-medoid clustering. While diversity *only* looks at the elements in the chosen subset, representativeness also worries about their similarity with the remaining elements in the superset. Denote :math:`s_{ij}` as the similarity between data points :math:`i` and :math:`j`. We can then define 

	.. math::
			f(X) = \\sum_{i \\in V} \\max_{j \\in X} s_{ij} 
	
	For each data point :math:`i` in the ground set :math:`V`, we compute the representative from subset :math:`X` which is closest to :math:`i` and add these similarities for all data points. 

	In a more generic setting, the set whose representation is desired (we call it represented set :math:`U`) may be different from the set whose subset is desired (we call it ground set :math:`V`). The expression for Facility-Location function then becomes

	.. math::
			f(X) = \\sum_{i \\in U} \\max_{j \\in X} s_{ij} 

	An alternative clustered implementation of Facility Location assumes a clustering of all ground set items and then the function value is computed over the clusters as 

	.. math::
			f(X) = \\sum_{l \\in {1....k}} \\sum_{i \\in C_l} \\max_{j \\in X \\cap C_l} s_{ij} 

	Facility-Location is monotone submodular.
	
	.. note:: 
		This function requires computing a :math:`\\mathcal{O}(n^2)` similarity function. However, as shown in :cite:`wei2014fast`, we can approximate this with a nearest neighbor graph, which will require much less storage, and also can run much faster for large ground set sizes.

	Parameters
	----------

	n : int
		Number of elements in the ground set, must be > 0.

	mode : string
		Can be "dense", "sparse" or "clustered". It specifies whether the Facility Location function should operate in dense mode (using a dense similarity kernel) or sparse mode (using a sparse similarity kernel) or clustered mode (evaluating over clusters).
	
	separate_rep: bool, optional
		Specifies whether a set different from ground set should be used as represented set (whose representation is desired).
	
	n_rep : int, optional
		Number of elements in the represented set if separate_rep=True.
	
	sijs : numpy.ndarray or scipy.sparse.csr.csr_matrix, optional
		When separate_rep=False, this is the similarity kernel (dense or sparse) between the elements of the ground set, to be used for getting :math:`s_{ij}` entries as defined above. Shape of dense kernel in this case must be n X n. When separate_rep=True, mode must be "dense" and this is the dense similarity kernel between the represented set and the ground set. Shape in this case must be n_rep X n. When sijs is not provided, it is computed internally in C++ based on the following additional parameters.

	data : numpy.ndarray, optional
		Matrix of shape n X num_features containing the ground set data elements. data[i] should contain the num-features dimensional features of element i. Used to compute the similarity kernel. It is optional (and is ignored if provided) if sijs has been provided.

	data_rep : numpy.ndarray, optional
		Represented set data matrix (used to compute the dense similarity kernel) if separate_rep=True and when a similarity kernel is not provided.

	num_clusters : int, optional
		Number of clusters in the ground set. Used only if mode = "clustered". Must be provided if cluster_labels is provided. If cluster_labels is not provided, clusters will be created using sklearn's BIRCH method. In this case if num_clusters is not provided, BIRCH will produce an optimum number of clusters.

	cluster_labels : list, optional
		List of size n that contains cluster label for each item in the groundset. If mode=clustered and cluster_labels is not provided, clustering is done internally using sklearn's BIRCH.

	metric : str, optional
		Similarity metric to be used for computing the similarity kernel. Can be "cosine" for cosine similarity or "euclidean" for similarity based on euclidean distance. Default is "cosine".
	
	num_neighbors : int, optional
		Number of neighbors applicable for the sparse similarity kernel. Must not be provided if mode is "dense". Must be provided if either a sparse kernel is provided or is to be computed.
	
	create_dense_cpp_kernel_in_python : bool, optional
		Should be set to False ONLY when a similarity kernel is not provided and a CPP kernel is desired to be created in CPP. Default is True.
	
	pybind_mode : string, optional
		Specifies mode of pybind type conversion from Python to CPP. Can be one of *list*, *numpyarray* and *array*. *list* is the slowest, requiring converting numpy arrays to Python lists. *numpyarray" relies on automatic conversion. *array* leverages native data types and is the fastest. Default is "array".

	"""

	#@profile
	def __init__(self, n, mode, separate_rep=None, n_rep=None, sijs=None, data=None, data_rep=None, num_clusters=None, cluster_labels=None, metric="cosine", num_neighbors=None, create_dense_cpp_kernel_in_python=True, pybind_mode="array"):
		self.n = n
		self.n_rep = n_rep
		self.mode = mode
		self.metric = metric
		self.sijs = sijs
		self.data = data
		self.data_rep=data_rep
		self.num_neighbors = num_neighbors
		#self.partial = partial
		#self.ground_sub = ground_sub
		self.separate_rep=separate_rep
		self.clusters=None
		self.cluster_sijs=None
		self.cluster_map=None
		self.cluster_labels=cluster_labels
		self.num_clusters=num_clusters
		self.cpp_obj = None
		self.cpp_sijs = None
		self.cpp_ground_sub = None
		self.cpp_content = None
		self.effective_ground = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		# if self.partial==True:
		# 	if type(self.ground_sub) == type(None) or len(self.ground_sub) == 0:
		# 		raise Exception("ERROR: Restricted subset of ground set not specified or empty for partial mode")
		# 	if self.mode == "clustered" or mode == "sparse":
		# 		raise Exception("clustered or sparse mode not supported if partial = True")
		# 	if not all(ele >= 0 and ele <= self.n-1 for ele in self.ground_sub):
		# 		raise Exception("Restricted subset of ground set contains invalid values")
		# 	if self.separate_rep == True:
		# 		raise Exception("Partial not supported if separate_rep = True")
		
		if self.mode not in ['dense', 'sparse', 'clustered']:
			raise Exception("ERROR: Incorrect mode. Must be one of 'dense', 'sparse' or 'clustered'")
		
		# if self.metric not in ['euclidean', 'cosine']:
		# 	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")

		if self.separate_rep == True:
			if self.n_rep  is None or self.n_rep <=0:
				raise Exception("ERROR: separate represented intended but number of elements in represented not specified or not positive")	
			if self.mode != "dense":
				raise Exception("Only dense mode supported if separate_rep = True")
			
		if self.mode == "clustered":
			if type(self.cluster_labels) != type(None) and (self.num_clusters  is None or self.num_clusters <= 0):
				raise Exception("ERROR: Positive number of clusters must be provided in clustered mode when cluster_labels is provided")
			# if self.cluster_labels  is None or len(cluster_labels) != self.n:
			# 	raise Exception("ERROR: Cluster ID/label for each element in the ground set is needed")
			if type(self.cluster_labels) == type(None) and self.num_clusters is not None and self.num_clusters <= 0:
				raise Exception("Invalid number of clusters provided") 
			if type(self.cluster_labels) != type(None) and len(self.cluster_labels) != self.n:
				raise Exception("ERROR: cluster_labels's size is NOT same as ground set size")
			if type(self.cluster_labels) != type(None) and not all(ele >= 0 and ele <= self.num_clusters-1 for ele in self.cluster_labels):
				raise Exception("Cluster IDs/labels contain invalid values")

		if type(self.sijs) != type(None): # User has provided similarity kernel
			if create_dense_cpp_kernel_in_python == False:
				raise Exception("ERROR: create_dense_cpp_kernel_in_python is to be set to False ONLY when a similarity kernel is not provided and a CPP kernel is desired to be created in CPP")
			if type(self.sijs) == scipy.sparse.csr.csr_matrix:
				if num_neighbors is None or num_neighbors <= 0:
					raise Exception("ERROR: Positive num_neighbors must be provided for given sparse kernel")
				if mode != "sparse":
					raise Exception("ERROR: Sparse kernel provided, but mode is not sparse")
			elif type(self.sijs) == np.ndarray:
				if self.separate_rep is None:
					raise Exception("ERROR: separate_rep bool must be specified with custom dense kernel")
				if mode != "dense":
					raise Exception("ERROR: Dense kernel provided, but mode is not dense")
			else:
				raise Exception("Invalid kernel provided")
			#TODO: is the below dimensionality check valid for both dense and sparse kernels?
			if self.separate_rep == True:
				if np.shape(self.sijs)[1]!=self.n or np.shape(self.sijs)[0]!=self.n_rep:
					raise Exception("ERROR: Inconsistency between n_rep, n and no of rows, columns of given kernel")
			else:
				if np.shape(self.sijs)[0]!=self.n or np.shape(self.sijs)[1]!=self.n:
					raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity kernel")
			if type(self.data) != type(None) or type(self.data_rep) != type(None):
				print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
		else: #similarity kernel has not been provided
			if type(self.data) != type(None): 
				if self.separate_rep == True:
					if type(self.data_rep) == type(None):
						raise Exception("Represented data matrix not given")
					if np.shape(self.data)[0]!=self.n or np.shape(self.data_rep)[0]!=self.n_rep:
						raise Exception("ERROR: Inconsistentcy between n, n_rep and no of examples in the given ground data matrix and represented data matrix")
				else:
					if type(self.data_rep) != type(None):
						print("WARNING: Represented data matrix not required but given, will be ignored.")
					if np.shape(self.data)[0]!=self.n:
						raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")
				
				if self.mode=="clustered":
					self.clusters, self.cluster_sijs, self.cluster_map = create_cluster_kernels(self.data.tolist(), self.metric, self.cluster_labels, self.num_clusters) #creates clusters if not provided
				else:
					if self.separate_rep==True: #mode in this case will always be dense
						if create_dense_cpp_kernel_in_python == True:
							self.sijs = np.array(subcp.create_kernel_NS(self.data.tolist(),self.data_rep.tolist(), self.metric))
					else:
						if self.mode == "dense":
							if self.num_neighbors  is not None:
								raise Exception("num_neighbors wrongly provided for dense mode")
							#self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
							if create_dense_cpp_kernel_in_python == True:
								self.sijs = np.array(subcp.create_square_kernel_dense(self.data.tolist(), self.metric))
						else:
							self.cpp_content = np.array(subcp.create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
							val = self.cpp_content[0]
							#TODO: these two lambdas take quite a bit of time, worth optimizing
							#row = list(map(lambda arg: int(arg), self.cpp_content[1]))
							#col = list(map(lambda arg: int(arg), self.cpp_content[2]))
							# row = [int(x) for x in self.cpp_content[1]]
							# col = [int(x) for x in self.cpp_content[2]]
							row = list(self.cpp_content[1].astype(int))
							col = list(self.cpp_content[2].astype(int))
							self.sijs = sparse.csr_matrix((val, (row, col)), [n,n])
			else:
				raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")
		
		# if self.partial==None:
		# 	self.partial = False
		
		if separate_rep==None:
			self.separate_rep = False
		
		# if self.partial==False: 
		self.cpp_ground_sub = {-1} #Provide a dummy set for pybind11 binding to be successful
		
		#Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
		if self.mode=="dense" and create_dense_cpp_kernel_in_python == True:
			if pybind_mode == "list":
				self.cpp_sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure
				
				if type(self.cpp_sijs[0])==int or type(self.cpp_sijs[0])==float: #Its critical that we pass a list of list to pybind11
																				#This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
					l=[]
					l.append(self.cpp_sijs)
					self.cpp_sijs=l

				self.cpp_obj = FacilityLocation(self.n, self.cpp_sijs, False, self.cpp_ground_sub, self.separate_rep)
			# elif pybind_mode == "memoryview":
			# 	self.cpp_obj = FacilityLocation(self.n, memoryview(self.sijs), False, self.cpp_ground_sub, self.separate_rep)
			elif pybind_mode == "numpyarray":
				self.cpp_obj = FacilityLocation(self.n, self.sijs, False, self.cpp_ground_sub, self.separate_rep)
			elif pybind_mode == "array32":
				# print("Kernel's type = ", self.sijs.dtype)
				self.sijs.astype('float32', copy=False)
				#self.cpp_obj = FacilityLocation2(self.n, self.sijs, False, {-1}, False);
				self.cpp_obj = FacilityLocation2()
				self.cpp_obj.pybind_init(self.n, self.sijs, False, {-1}, self.separate_rep);
			elif pybind_mode == "array64":
				# print("Kernel's type = ", self.sijs.dtype)
				self.sijs.astype('float64', copy=False)
				#self.cpp_obj = FacilityLocation2(self.n, self.sijs, False, {-1}, False);
				self.cpp_obj = FacilityLocation2()
				self.cpp_obj.pybind_init(self.n, self.sijs, False, {-1}, self.separate_rep);
			elif pybind_mode == "array":
				#self.cpp_obj = FacilityLocation2(self.n, self.sijs, False, {-1}, False);
				self.cpp_obj = FacilityLocation2()
				self.cpp_obj.pybind_init(self.n, self.sijs, False, {-1}, self.separate_rep);
			else:
				raise Exception("Invalid pybind mode!")
		
		elif self.mode=="dense" and create_dense_cpp_kernel_in_python == False:
			if self.separate_rep == True:
				self.cpp_obj = FacilityLocation(self.n, self.data.tolist(), self.data_rep.tolist(), True, self.metric)
			else:
				self.cpp_obj = FacilityLocation(self.n, self.data.tolist(), [[0.]], False, self.metric)
		
		elif self.mode=="sparse": #break scipy sparse matrix to native component lists (for csr implementation)
			self.cpp_sijs = {}
			self.cpp_sijs['arr_val'] = self.sijs.data.tolist() #contains non-zero values in matrix (row major traversal)
			self.cpp_sijs['arr_count'] = self.sijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
			self.cpp_sijs['arr_col'] = self.sijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
			self.cpp_obj = FacilityLocation(self.n, self.cpp_sijs['arr_val'], self.cpp_sijs['arr_count'], self.cpp_sijs['arr_col'])
		
		elif self.mode=="clustered":
			l_temp = []
			#TODO: this for loop can be optimized
			for el in self.cluster_sijs:
				temp=el.tolist()
				if type(temp[0])==int or type(temp[0])==float: 
					l=[]
					l.append(temp)
					temp=l
				l_temp.append(temp)
			self.cluster_sijs = l_temp

			self.cpp_obj = FacilityLocation(self.n, self.clusters, self.cluster_sijs, self.cluster_map)

		#self.cpp_ground_sub=self.cpp_obj.getEffectiveGroundSet()
		#self.ground_sub=self.cpp_ground_sub
		self.effective_ground = self.cpp_obj.getEffectiveGroundSet()