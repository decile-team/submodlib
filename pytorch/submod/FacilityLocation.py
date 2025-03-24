import numpy as np
import scipy
from scipy import sparse
from helper import *
from ..SetFunction import SetFunction

class FacilityLocationFunction(SetFunction):
    def __init__(self, n, mode, separate_rep=None, n_rep=None, sijs=None, data=None, data_rep=None, num_clusters=None, cluster_labels=None, metric="cosine", num_neighbors=None,
                 dense_kernel = None, data_master = None, create_dense_cpp_kernel_in_python = True, partial = False, seperate_master = False):
        self.n = n
        self.n_rep = n_rep
        self.mode = mode
        self.metric = metric
        self.sijs = sijs
        self.data = data
        self.partial = partial
        self.data_rep = data_rep
        self.num_neighbors = num_neighbors
        self.separate_rep = separate_rep
        self.clusters = None
        self.cluster_sijs = None
        self.cluster_map = None
        self.cluster_labels = cluster_labels
        self.num_clusters = num_clusters
        self.cpp_obj = None
        self.cpp_sijs = None
        self.cpp_ground_sub = None
        self.cpp_content = None
        self.effective_ground = None
        self.seperate_master = seperate_master
        self.dense_kernel = dense_kernel
        self.data_master = data_master

        if self.n <= 0:
            raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse', 'clustered']:
            raise Exception("ERROR: Incorrect mode. Must be one of 'dense', 'sparse' or 'clustered'")

        if self.separate_rep == True:
            if self.n_rep is None or self.n_rep <= 0:
                raise Exception("ERROR: separate represented intended but number of elements in represented not specified or not positive")
            if self.mode != "dense":
                raise Exception("Only dense mode supported if separate_rep = True")

        if self.mode == "clustered":
            if type(self.cluster_labels) != type(None) and (self.num_clusters  is None or self.num_clusters <= 0):
                raise Exception("ERROR: Positive number of clusters must be provided in clustered mode when cluster_labels is provided")
            if type(self.cluster_labels) == type(None) and self.num_clusters is not None and self.num_clusters <= 0:
                raise Exception("Invalid number of clusters provided")
            if type(self.cluster_labels) != type(None) and len(self.cluster_labels) != self.n:
                raise Exception("ERROR: cluster_labels's size is NOT same as ground set size")
            if type(self.cluster_labels) != type(None) and not all(ele >= 0 and ele <= self.num_clusters-1 for ele in self.cluster_labels):
                raise Exception("Cluster IDs/labels contain invalid values")

        if type(self.sijs) != type(None):
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

            if self.separate_rep == True:
                if np.shape(self.sijs)[1] != self.n or np.shape(self.sijs)[0] != self.n_rep:
                    raise Exception("ERROR: Inconsistency between n_rep, n and no of rows, columns of given kernel")
            else:
                if np.shape(self.sijs)[0] != self.n or np.shape(self.sijs)[1] != self.n:
                    raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity kernel")

            if type(self.data) != type(None) or type(self.data_rep) != type(None):
                print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
        else:
            if type(self.data) != type(None):
                if self.separate_rep == True:
                    if type(self.data_rep) == type(None):
                        raise Exception("Represented data matrix not given")
                    if np.shape(self.data)[0] != self.n or np.shape(self.data_rep)[0] != self.n_rep:
                        raise Exception("ERROR: Inconsistentcy between n, n_rep and no of examples in the given ground data matrix and represented data matrix")
                else:
                    if type(self.data_rep) != type(None):
                        print("WARNING: Represented data matrix not required but given, will be ignored.")
                    if np.shape(self.data)[0] != self.n:
                        raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")

                if self.mode == "clustered":
                    self.clusters, self.cluster_sijs, self.cluster_map = create_cluster_kernels(self.data.tolist(), self.metric, self.cluster_labels, self.num_clusters)
                else:
                    if self.separate_rep == True:
                        if create_dense_cpp_kernel_in_python == True:
                            self.sijs = np.array(create_kernel_NS(self.data.tolist(), self.data_rep.tolist(), self.metric))
                    else:
                        if self.mode == "dense":
                            if self.num_neighbors is not None:
                                raise Exception("num_neighbors wrongly provided for dense mode")
                            if create_dense_cpp_kernel_in_python == True:
                                pass
                                # self.sijs = np.array(create_square_kernel_dense(self.data.tolist(), self.metric))
                        else:
                            self.cpp_content = np.array(create_kernel(self.data.tolist(), self.metric, self.num_neighbors))
                            val = self.cpp_content[0]
                            row = list(self.cpp_content[1].astype(int))
                            col = list(self.cpp_content[2].astype(int))
                            self.sijs = sparse.csr_matrix((val, (row, col)), [n,n])
            else:
                raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")

        # self.cpp_ground_sub = {-1}

        if separate_rep == None:
            self.separate_rep = False

        elif self.mode == "sparse":
            self.cpp_sijs = {}
            self.cpp_sijs["arr_val"] = self.sijs.data.tolist()
            self.cpp_sijs["arr_count"] = self.sijs.indptr.tolist()
            self.cpp_sijs["arr_col"] = self.sijs.indices.tolist()
            # self.cpp_obj = FacilityLocation(self.n, self.cpp_sijs["arr_val"], self.cpp_sijs["arr_count"], self.cpp_sijs["arr_col"])
        elif self.mode == "clustered":
            l_temp = []
            for el in self.cluster_sijs:
                temp = el.tolist()
                if isinstance(temp[0], int) or isinstance(temp[0], float):
                    l = []
                    l.append(temp)
                    temp = l
                l_temp.append(temp)
            self.cluster_sijs = l_temp


        if self.mode == 'dense':
          if self.dense_kernel == None:
            self.dense_constructor_no_kernel(n = self.n, data = self.data, data_master = self.data_master) ## dense mode with no dense_kernel
          elif self.dense_kernel != None:
            self.dense_constructor(n = self.n, dense_kernel = self.dense_kernel, ground = self.data, partial = self.partial, separate_master = self.separate_master) ## dense mode with dense_kernel
        ### other modes are remaining
        elif self.mode == 'sparse':
          pass
        elif self.mode == 'clustered':
          pass

        self.effective_ground = self.get_effective_ground_set()


    def dense_constructor(self, n, dense_kernel, partial = False, ground = None, separate_master = False):
        self.n = n
        self.mode = 'dense'
        self.dense_kernel = dense_kernel
        self.partial = partial
        self.separate_master = separate_master

        if partial:
            self.effective_ground_set = ground
        else:
            self.effective_ground_set = set(range(n))

        self.num_effective_groundset = len(self.effective_ground_set)

        if separate_master:
            self.n_master = len(dense_kernel)
            self.master_set = set(range(self.n_master))
        else:
            self.n_master = self.num_effective_groundset
            self.master_set = self.effective_ground_set

        self.similarity_with_nearest_in_effective_x = np.zeros(self.n_master)

        if partial:
            self.original_to_partial_index_map = {val: i for i, val in enumerate(self.effective_ground_set)}

    # Constructor for dense mode (kernel not supplied)
    def dense_constructor_no_kernel(self, n, data, data_master, separate_master = False, metric = 'cosine'):
        if separate_master:
            self.dense_kernel = create_kernel_NS(data, data_master, metric)
        else:
            self.dense_kernel = create_square_kernel_dense(data, metric)

        self.mode = 'dense'
        self.partial = False

        self.n = n
        self.separate_master = separate_master

        self.effective_ground_set = set(range(n))
        self.num_effective_groundset = n

        if separate_master:
            self.n_master = len(self.dense_kernel)
            self.master_set = set(range(self.n_master))
        else:
            self.n_master = n
            self.master_set = self.effective_ground_set

        self.similarity_with_nearest_in_effective_x = np.zeros(self.n_master)

    # Constructor for sparse mode
    def sparse_constructor(self, n, arr_val, arr_count, arr_col):
        self.n = n
        self.mode = 'sparse'
        self.partial = False
        self.separate_master = False

        self.sparse_kernel = self.SparseSim(arr_val, arr_count, arr_col)

        self.effective_ground_set = set(range(n))
        self.num_effective_groundset = n

        self.n_master = self.num_effective_groundset
        self.master_set = self.effective_ground_set

        self.similarity_with_nearest_in_effective_x = np.zeros(self.n_master)

    # Constructor for cluster mode
    def cluster_constructor(self, n, clusters, cluster_kernels, cluster_index_map):
        self.n = n
        self.mode = 'clustered'
        self.num_clusters = len(clusters)
        self.clusters = clusters
        self.cluster_kernels = cluster_kernels
        self.cluster_index_map = cluster_index_map
        self.partial = False
        self.separate_master = False

        self.effective_ground_set = set(range(n))
        self.num_effective_groundset = n

        self.n_master = self.num_effective_groundset
        self.master_set = self.effective_ground_set

        self.cluster_ids = [0] * n
        for i, ci in enumerate(clusters):
            for ind in ci:
                self.cluster_ids[ind] = i

        self.relevant_x = [[] for _ in range(self.num_clusters)]
        self.clustered_similarity_with_nearest_in_relevant_x = np.zeros(n)

    # def clone(self):
    #     return FacilityLocation(**self.__dict__)

    def evaluate(self, X):
        effective_X = X.intersection(self.effective_ground_set) if self.partial else X
        result = 0

        if effective_X:
            if self.mode == 'dense':
                for ind in self.master_set:
                    result += self.get_max_sim_dense(ind, effective_X)
            elif self.mode == 'sparse':
                for ind in self.master_set:
                    result += self.get_max_sim_sparse(ind, effective_X)
            else:  # clustered
                for i in range(self.num_clusters):
                    relevant_subset = X.intersection(self.clusters[i])
                    if relevant_subset:
                        for ind in self.clusters[i]:
                            result += self.get_max_sim_cluster(ind, relevant_subset, i)

        return result

    def evaluate_with_memoization(self, X):
        effective_X = X.intersection(self.effective_ground_set) if self.partial else X
        result = 0

        if effective_X:
            if self.mode == 'dense' or self.mode == 'sparse':
                for ind in self.master_set:
                    result += self.similarity_with_nearest_in_effective_x[ind]
            else:  # clustered
                for i in range(self.num_clusters):
                    if self.relevant_x[i]:
                        for ind in self.clusters[i]:
                            result += self.clustered_similarity_with_nearest_in_relevant_x[ind]

        return result

    def marginal_gain(self, X, item):
        effective_X = X.intersection(self.effective_ground_set) if self.partial else X
        gain = 0

        if item not in effective_X:
            if self.mode == 'dense':
                print(self.master_set)
                for ind in self.master_set:
                    m = self.get_max_sim_dense(ind, effective_X)
                    if self.dense_kernel[item][ind] > m:
                        m = self.dense_kernel[item][ind]
                    gain += m - self.similarity_with_nearest_in_effective_x[ind]
            elif self.mode == 'sparse':
                for ind in self.master_set:
                    m = self.get_max_sim_sparse(ind, effective_X)
                    if self.sparse_kernel[item, ind] > m:
                        m = self.sparse_kernel[item, ind]
                    gain += m - self.similarity_with_nearest_in_effective_x[ind]
            else:  # clustered
                cluster_id = self.cluster_ids[item]
                relevant_subset = effective_X.intersection(self.clusters[cluster_id])
                for ind in self.clusters[cluster_id]:
                    m = self.get_max_sim_cluster(ind, relevant_subset, cluster_id)
                    if self.cluster_kernels[cluster_id][item][ind] > m:
                        m = self.cluster_kernels[cluster_id][item][ind]
                    gain += m - self.clustered_similarity_with_nearest_in_relevant_x[ind]

        return gain
    def marginal_gain_with_memoization(self, X, item, enable_checks):
      effective_X = set()
      gain = 0

      if self.partial:
          effective_X = X.intersection(self.effective_ground_set)
      else:
          effective_X = X

      if enable_checks and item in effective_X:
          return 0

      if self.partial and item not in self.effective_ground_set:
          return 0

      if self.mode == 'dense':
          for ind in self.master_set:
              if self.partial:
                  if self.dense_kernel[ind][item] > self.similarity_with_nearest_in_effective_x[self.original_to_partial_index_map[ind]]:
                      gain += self.dense_kernel[ind][item] - self.similarity_with_nearest_in_effective_x[self.original_to_partial_index_map[ind]]
              else:
                  if self.dense_kernel[ind][item] > self.similarity_with_nearest_in_effective_x[ind]:
                      gain += self.dense_kernel[ind][item] - self.similarity_with_nearest_in_effective_x[ind]
      elif self.mode == 'sparse':
          for ind in self.master_set:
              temp = self.sparse_kernel[ind, item]
              if temp > self.similarity_with_nearest_in_effective_x[ind]:
                  gain += temp - self.similarity_with_nearest_in_effective_x[ind]
      else:  # clustered
          i = self.cluster_ids[item]
          item_ = self.cluster_index_map[item]
          relevant_subset = self.relevant_x[i]
          ci = self.clusters[i]

          if len(relevant_subset) == 0:
              for ind in ci:
                  ind_ = self.cluster_index_map[ind]
                  gain += self.cluster_kernels[i][ind_][item_]
          else:
              for ind in ci:
                  ind_ = self.cluster_index_map[ind]
                  if self.cluster_kernels[i][ind_][item_] > self.clustered_similarity_with_nearest_in_relevant_x[ind]:
                      gain += self.cluster_kernels[i][ind_][item_] - self.clustered_similarity_with_nearest_in_relevant_x[ind]

      return gain


    def update_memoization(self, X, item):
        effective_X = set()

        if self.partial:
            effective_X = X.intersection(self.effective_ground_set)
        else:
            effective_X = X

        if item in effective_X:
            return

        if self.partial and item not in self.effective_ground_set:
            return

        if self.mode == 'dense':
            for ind in self.master_set:
                if self.partial:
                    if self.dense_kernel[ind][item] > self.similarity_with_nearest_in_effective_x[self.original_to_partial_index_map[ind]]:
                        self.similarity_with_nearest_in_effective_x[self.original_to_partial_index_map[ind]] = self.dense_kernel[ind][item]
                else:
                    if self.dense_kernel[ind][item] > self.similarity_with_nearest_in_effective_x[ind]:
                        self.similarity_with_nearest_in_effective_x[ind] = self.dense_kernel[ind][item]
        elif self.mode == 'sparse':
            for ind in self.master_set:
                temp_val = self.sparse_kernel[ind, item]
                if temp_val > self.similarity_with_nearest_in_effective_x[ind]:
                    self.similarity_with_nearest_in_effective_x[ind] = temp_val
        else:  # clustered
            i = self.cluster_ids[item]
            item_ = self.cluster_index_map[item]
            ci = self.clusters[i]

            for ind in ci:
                ind_ = self.cluster_index_map[ind]
                if self.cluster_kernels[i][ind_][item_] > self.clustered_similarity_with_nearest_in_relevant_x[ind]:
                    self.clustered_similarity_with_nearest_in_relevant_x[ind] = self.cluster_kernels[i][ind_][item_]

            self.relevant_x[i].add(item)


    def get_effective_ground_set(self):
        return set(range(self.n))


    def cluster_init(self, n_, dense_kernel_, ground_, partial, lambda_):
        self.n = n_
        self.partial = partial
        self.effective_ground_set = ground_
        self.n_master = len(dense_kernel_)
        self.master_set = set(range(self.n_master))
        self.similarity_with_nearest_in_effective_x = np.zeros(self.n_master)
        self.mode = 'dense'
        self.dense_kernel = dense_kernel_
        self.original_to_partial_index_map = {val: i for i, val in enumerate(self.effective_ground_set)}
        self.clustered_similarity_with_nearest_in_relevant_x = np.zeros(n_)
        self.relevant_x = [set() for _ in range(n_)]


    def clear_memoization(self):
        if self.mode == 'dense' or self.mode == 'sparse':
            self.similarity_with_nearest_in_effective_x = np.zeros(self.n_master)
        else:
            self.relevant_x = [set() for _ in range(self.num_clusters)]
            self.clustered_similarity_with_nearest_in_relevant_x = np.zeros(self.n)


    def set_memoization(self, X):
        self.clear_memoization()
        temp = set()
        for elem in X:
            self.update_memoization(temp, elem)
            temp.add(elem)
