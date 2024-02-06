from typing import List, Set
import random
from helper import *
from ..SetFunction import SetFunction

class GraphCutFunction(SetFunction):
    def __init__(self, n, mode, lambdaVal, separate_rep=None, n_rep=None, mgsijs=None, ggsijs=None, data=None, data_rep=None, metric="cosine", num_neighbors=None,
                 master_ground_kernel: List[List[float]] = None,
                 ground_ground_kernel: List[List[float]] = None, arr_val: List[float] = None,
                 arr_count: List[int] = None, arr_col: List[int] = None, partial: bool = False,
                 ground: Set[int] = None):
        super(SetFunction, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n = n
        self.mode = mode
        self.lambda_ = lambdaVal
        self.separate_rep=separate_rep
        self.n_rep = n_rep
        self.partial = partial
        self.original_to_partial_index_map = {}
        self.mgsijs = mgsijs
        self.ggsijs = ggsijs
        self.data = data
        self.data_rep=data_rep
        self.metric = metric
        self.num_neighbors = num_neighbors
        self.effective_ground_set = set(range(n))
        self.clusters=None
        self.cluster_sijs=None
        self.cluster_map=None
        self.ggsijs = None
        self.mgsijs = None
        self.content = None
        self.effective_ground = None

        if self.n <= 0:
          raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse']:
          raise Exception("ERROR: Incorrect mode. Must be one of 'dense' or 'sparse'")
        if self.separate_rep == True:
          if self.n_rep is None or self.n_rep <=0:
            raise Exception("ERROR: separate represented intended but number of elements in represented not specified or not positive")
          if self.mode != "dense":
            raise Exception("Only dense mode supported if separate_rep = True")
          if (type(self.mgsijs) != type(None)) and (type(self.mgsijs) != np.ndarray):
            raise Exception("mgsijs provided, but is not dense")
          if (type(self.ggsijs) != type(None)) and (type(self.ggsijs) != np.ndarray):
            raise Exception("ggsijs provided, but is not dense")

        if mode == "dense":
            self.master_ground_kernel = master_ground_kernel
            self.ground_ground_kernel = ground_ground_kernel

            if ground_ground_kernel is not None:
                self.separate_master = True

            if partial:
                self.effective_ground_set = ground
            else:
                self.effective_ground_set = set(range(n))

            self.num_effective_ground_set = len(self.effective_ground_set)

            self.n_master = self.num_effective_ground_set
            self.master_set = self.effective_ground_set

            if partial:
                self.original_to_partial_index_map = {elem: ind for ind, elem in enumerate(self.effective_ground_set)}

            self.total_similarity_with_subset = [random.random() for _ in range(self.num_effective_ground_set)]
            self.total_similarity_with_master = [random.random() for _ in range(self.num_effective_ground_set)]
            self.master_ground_kernel = [[random.random() for _ in range(self.num_effective_ground_set)] for _ in range(self.num_effective_ground_set)]
            self.ground_ground_kernel = [[random.random() for _ in range(self.num_effective_ground_set)] for _ in range(self.num_effective_ground_set)]
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if partial else elem
                self.total_similarity_with_subset[index] = 1
                self.total_similarity_with_master[index] = 1
                for j in self.master_set:
                    self.total_similarity_with_master[index] += self.master_ground_kernel[j][elem]

            if self.separate_rep == True:
              if type(self.mgsijs) == type(None):
                #not provided mgsij - make it
                if (type(data) == type(None)) or (type(data_rep) == type(None)):
                  raise Exception("Data missing to compute mgsijs")
                if np.shape(self.data)[0]!=self.n or np.shape(self.data_rep)[0]!=self.n_rep:
                  raise Exception("ERROR: Inconsistentcy between n, n_rep and no of examples in the given ground data matrix and represented data matrix")

                #create_kernel_NS is there .................... find it and define it not found in helper.py but used as here
                # self.mgsijs = np.array(subcp.create_kernel_NS(self.data.tolist(),self.data_rep.tolist(), self.metric))
              else:
                #provided mgsijs - verify it's dimensionality
                if np.shape(self.mgsijs)[1]!=self.n or np.shape(self.mgsijs)[0]!=self.n_rep:
                  raise Exception("ERROR: Inconsistency between n_rep, n and no of rows, columns of given mg kernel")
              if type(self.ggsijs) == type(None):
                #not provided ggsijs - make it
                if type(data) == type(None):
                  raise Exception("Data missing to compute ggsijs")
                if self.num_neighbors is not None:
                  raise Exception("num_neighbors wrongly provided for dense mode")
                self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
                self.content = np.array(create_kernel(X = torch.tensor(self.data), metric = self.metric, num_neigh = self.num_neighbors).to_dense())
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
                self.content = np.array(create_kernel(X = torch.tensor(self.data), metric = self.metric, num_neigh = self.num_neighbors).to_dense())
                val = self.content[0]
                row = list(self.content[1].astype(int))
                col = list(self.content[2].astype(int))
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
                raise Exception("Two kernels have been wrongly provided when separate_rep=False")
        elif mode == "sparse":
            if self.separate_rep == True:
                raise Exception("Separate represented is supported only in dense mode")
            if self.num_neighbors is None or self.num_neighbors <=0:
              raise Exception("Valid num_neighbors is needed for sparse mode")
            if (type(self.ggsijs) == type(None)) and (type(self.mgsijs) == type(None)):
              #no kernel is provided make ggsij sparse kernel
              if type(data) == type(None):
                raise Exception("Data missing to compute ggsijs")
              self.content = np.array(create_kernel(X = torch.tensor(self.data), metric = self.metric, num_neigh = self.num_neighbors).to_dense())
              val = self.content[0]
              row = list(self.content[1].astype(int))
              col = list(self.content[2].astype(int))
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
              raise Exception("Two kernels have been wrongly provided when separate_rep=False")

        if self.separate_rep==None:
            self.separate_rep = False

        if self.mode=="dense" and self.separate_rep == False :
            self.ggsijs = self.ggsijs.tolist() #break numpy ndarray to native list of list datastructure

            if type(self.ggsijs[0])==int or type(self.ggsijs[0])==float: #Its critical that we pass a list of list to pybind11
                                            #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
              l=[]
              l.append(self.ggsijs)
              self.ggsijs=l

        elif self.mode=="dense" and self.separate_rep == True :
            self.ggsijs = self.ggsijs.tolist() #break numpy ndarray to native list of list datastructure

            if type(self.ggsijs[0])==int or type(self.ggsijs[0])==float: #Its critical that we pass a list of list to pybind11
                                            #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
              l=[]
              l.append(self.ggsijs)
              self.ggsijs=l

            self.mgsijs = self.mgsijs.tolist() #break numpy ndarray to native list of list datastructure

            if type(self.mgsijs[0])==int or type(self.mgsijs[0])==float: #Its critical that we pass a list of list to pybind11
                                            #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
              l=[]
              l.append(self.mgsijs)
              self.mgsijs=l

            # self.cpp_obj = GraphCutpy(self.n, self.cpp_mgsijs, self.cpp_ggsijs, self.lambdaVal)

        elif self.mode == "sparse":
            self.ggsijs = {}
            # self.ggsijs['arr_val'] = self.ggsijs.data.tolist() #contains non-zero values in matrix (row major traversal)
            # self.ggsijs['arr_count'] = self.ggsijs.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
            # self.ggsijs['arr_col'] = self.ggsijs.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
            # # self.cpp_obj = GraphCutpy(self.n, self.cpp_ggsijs['arr_val'], self.cpp_ggsijs['arr_count'], self.cpp_ggsijs['arr_col'], lambdaVal)
        else:
            raise Exception("Invalid")

        self.effective_ground = self.get_effective_ground_set()

        # if mode == "dense":

        # elif mode == "sparse":
        #     if not arr_val or not arr_count or not arr_col:
        #         raise ValueError("Error: Empty/Corrupt sparse similarity kernel")

        #     self.sparse_kernel = SparseSim(arr_val, arr_count, arr_col)

        #     self.effective_ground_set = set(range(n))
        #     self.num_effective_ground_set = len(self.effective_ground_set)

        #     self.n_master = self.num_effective_ground_set
        #     self.master_set = self.effective_ground_set

        #     self.total_similarity_with_subset = [0] * n
        #     self.total_similarity_with_master = [0] * n

        #     for i in range(n):
        #         self.total_similarity_with_subset[i] = 0
        #         self.total_similarity_with_master[i] = 0

        #         for j in range(n):
        #             self.total_similarity_with_master[i] += self.sparse_kernel.get_val(j, i)

        # else:
        #     raise ValueError("Invalid mode")

    def evaluate(self, X: Set[int]) -> float:
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if not effective_x:
            return 0

        result = 0

        if self.mode == "dense":
            for elem in effective_x:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                result += self.total_similarity_with_master[index]

                for elem2 in effective_x:
                    result -= self.lambda_ * self.ground_ground_kernel[elem][elem2]

        elif self.mode == "sparse":
            for elem in effective_x:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                result += self.total_similarity_with_master[index]

                for elem2 in effective_x:
                    result -= self.lambda_ * self.sparse_kernel.get_val(elem, elem2)

        return result

    def evaluate_with_memoization(self, X: Set[int]) -> float:
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if not effective_x:
            return 0

        result = 0

        if self.mode == "dense" or self.mode == "sparse":
            for elem in effective_x:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                result += self.total_similarity_with_master[index] - self.lambda_ * self.total_similarity_with_subset[index]

        return result

    def marginal_gain(self, X: Set[int], item: int) -> float:
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if item in effective_x or item not in self.effective_ground_set:
            return 0

        gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item]

        if self.mode == "dense":
            for elem in effective_x:
                gain -= 2 * self.lambda_ * self.ground_ground_kernel[item][elem]
            gain -= self.lambda_ * self.ground_ground_kernel[item][item]

        elif self.mode == "sparse":
            for elem in effective_x:
                gain -= 2 * self.lambda_ * self.sparse_kernel.get_val(item, elem)
            gain -= self.lambda_ * self.sparse_kernel.get_val(item, item)
        return gain

    # def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
    #     effective_x = X.intersection(self.effective_ground_set) if self.partial else X

    #     if enable_checks and item in effective_x:
    #         return 0

    #     if self.partial and item not in self.effective_ground_set:
    #         return 0

    #     gain = 0

    #     if self.mode == "dense":
    #         index = self.original_to_partial_index_map[item] if self.partial else item
    #         gain = self.total_similarity_with_master[index] - 2 * self.lambda_ * self.total_similarity_with_subset[index]
    #         gain = self.total_similarity_with_master[index] - 2 * self.lambda_ * self.total_similarity_with_subset[index] - self.lambda_ * self.ground_ground_kernel[item][item]

    #     elif self.mode == "sparse":
    #         index = self.original_to_partial_index_map[item] if self.partial else item
    #         gain = self.total_similarity_with_master[index] - 2 * self.lambda_ * self.total_similarity_with_subset[index] - self.lambda_ * self.sparse_kernel.get_val(item, item)

    #     return gain


    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool) -> float:
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
            gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item] \
                  - 2 * self.lambda_ * self.total_similarity_with_subset[self.original_to_partial_index_map[item] if self.partial else item] \
                  - self.lambda_ * self.ground_ground_kernel[item][item]
        elif self.mode == 'sparse':
            gain = self.total_similarity_with_master[self.original_to_partial_index_map[item] if self.partial else item] \
                  - 2 * self.lambda_ * self.total_similarity_with_subset[self.original_to_partial_index_map[item] if self.partial else item] \
                  - self.lambda_ * self.sparse_kernel.get_val(item, item)
        else:
            raise ValueError("Error: Only dense and sparse mode supported")
        # print("gain value",gain)
        return gain


    def update_memoization(self, X: Set[int], item: int):
        effective_x = X.intersection(self.effective_ground_set) if self.partial else X

        if item in effective_x or item not in self.effective_ground_set:
            return

        if self.mode == "dense":
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                # self.total_similarity_with_subset[index] += self.ground_ground_kernel[elem][item]

        elif self.mode == "sparse":
            for elem in self.effective_ground_set:
                index = self.original_to_partial_index_map[elem] if self.partial else elem
                self.total_similarity_with_subset[index] += self.sparse_kernel.get_val(elem, item)

    def get_effective_ground_set(self) -> Set[int]:
        return self.effective_ground_set

    def clear_memoization(self):
        if self.mode == "dense" or self.mode == "sparse":
            self.total_similarity_with_subset = [0] * self.num_effective_ground_set

    def set_memoization(self, X: Set[int]):
        temp = set()
        for elem in X:
            self.update_memoization(temp, elem)
            temp.add(elem)
