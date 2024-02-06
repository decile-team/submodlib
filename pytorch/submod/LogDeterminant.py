import math
from collections import defaultdict
import scipy
from helper import *
from ..SetFunction import SetFunction

class LogDeterminantFunction(SetFunction):

    def dot_product(self, x, y):
        return sum(xi * yi for xi, yi in zip(x, y))


    def __init__(self, n, mode, lambdaVal, arr_val=None, arr_count=None, arr_col=None, dense_kernel=None, partial=None,
                  sijs=None, data=None, metric="cosine", num_neighbors=None, memoizedC = None, memoizedD = None, data_master = None):
        self.n = n
        self.mode = mode
        self.metric = metric
        self.sijs = sijs
        self.data = data
        self.num_neighbors = num_neighbors
        self.lambdaVal = lambdaVal
        self.sijs = None
        self.content = None
        self.effective_ground = None
        self.partial = partial
        self.effective_ground_set = set(range(n))
        self.memoizedC = memoizedC
        self.memoizedD = memoizedD
        self.data_master = data_master
        self.dense_kernel = dense_kernel

        if self.n <= 0:
          raise Exception("ERROR: Number of elements in ground set must be positive")

        if self.mode not in ['dense', 'sparse', 'clustered']:
          raise Exception("ERROR: Incorrect mode. Must be one of 'dense', 'sparse' or 'clustered'")

        if self.metric not in ['euclidean', 'cosine']:
        	raise Exception("ERROR: Unsupported metric. Must be 'euclidean' or 'cosine'")
        if type(self.sijs) != type(None): # User has provided similarity kernel
          if type(self.sijs) == scipy.sparse.csr.csr_matrix:
            if num_neighbors is None or num_neighbors <= 0:
              raise Exception("ERROR: Positive num_neighbors must be provided for given sparse kernel")
            if mode != "sparse":
              raise Exception("ERROR: Sparse kernel provided, but mode is not sparse")
          elif type(self.sijs) == np.ndarray:
            if mode != "dense":
              raise Exception("ERROR: Dense kernel provided, but mode is not dense")
          else:
            raise Exception("Invalid kernel provided")
          #TODO: is the below dimensionality check valid for both dense and sparse kernels?
          if np.shape(self.sijs)[0]!=self.n or np.shape(self.sijs)[1]!=self.n:
            raise Exception("ERROR: Inconsistentcy between n and dimensionality of given similarity kernel")
          if type(self.data) != type(None):
            print("WARNING: similarity kernel found. Provided data matrix will be ignored.")
        else: #similarity kernel has not been provided
          if type(self.data) != type(None):
            if np.shape(self.data)[0]!=self.n:
              raise Exception("ERROR: Inconsistentcy between n and no of examples in the given data matrix")

            if self.mode == "dense":
              if self.num_neighbors  is not None:
                raise Exception("num_neighbors wrongly provided for dense mode")
              self.num_neighbors = np.shape(self.data)[0] #Using all data as num_neighbors in case of dense mode
            self.content = np.array(create_kernel( X = self.data.tolist(), metric = self.metric, mode = self.mode, num_neigh = self.num_neighbors))
            val = self.content[0]
            row = list(self.content[1].astype(int))
            col = list(self.content[2].astype(int))
            if self.mode=="dense":
              self.sijs = np.zeros((n,n))
              self.sijs[row,col] = val
            if self.mode=="sparse":
              self.sijs = sparse.csr_matrix((val, (row, col)), [n,n])
          else:
            raise Exception("ERROR: Neither ground set data matrix nor similarity kernel provided")


        #Breaking similarity matrix to simpler native data structures for implicit pybind11 binding
        if self.mode=="dense":
          self.sijs = self.sijs.tolist() #break numpy ndarray to native list of list datastructure

          if type(self.sijs[0])==int or type(self.sijs[0])==float: #Its critical that we pass a list of list to pybind11
                                          #This condition ensures the same in case of a 1D numpy array (for 1x1 sim matrix)
            l=[]
            l.append(self.sijs)
            self.sijs=l

        self.effective_ground = self.get_effective_ground_set()
        if self.mode == 'dense':
          if self.dense_kernel == None:
             self.dense_kernel = create_kernel_NS(X_ground = self.data, X_master = self.data, metric = self.metric)
          if self.partial:
            self.effectiveGroundSet = self.data
          else:
            self.effectiveGroundSet = set(range(n))
            self.numEffectiveGroundset = len(self.effectiveGroundSet)
            self.memoizedC = [[] for _ in range(self.numEffectiveGroundset)]
            self.prevDetVal = 0
            self.memoizedD = []
            self.prevItem = -1

            if self.partial:
                ind = 0
                for it in self.effectiveGroundSet:
                    self.originalToPartialIndexMap[it] = ind
                    ind += 1
                    self.memoizedD.append(np.sqrt(self.dense_kernel[it][it] + self.lambdaVal))
            else:
                for i in range(self.n):
                    self.memoizedD.append(np.sqrt(self.dense_kernel[i][i] + self.lambdaVal))

        elif arr_val is not None and arr_count is not None and arr_col is not None:
            self.n = n
            self.mode = 'sparse'
            self.lambdaVal = lambdaVal
            self.sparseKernel = SparseSim(arr_val, arr_count, arr_col)
            self.effectiveGroundSet = set(range(n_))
            self.numEffectiveGroundset = len(self.effectiveGroundSet)
            self.memoizedC = [[] for _ in range(n_)]
            self.memoizedD = []
            self.prevDetVal = 0
            self.prevItem = -1

            for i in range(self.n):
                self.memoizedD.append(np.sqrt(self.sparseKernel.get_val(i, i) + self.lambdaVal))

        else:
            raise ValueError("Invalid constructor arguments. Please provide either denseKernel or sparse kernel data.")

    def evaluate(self, X):
        currMemoizedC = self.memoizedC.copy()
        currMemoizedD = self.memoizedD.copy()
        currprevItem = self.prevItem
        currprevDetVal = self.prevDetVal
        self.setMemoization(X)
        result = self.evaluate_with_memoization(X)
        self.memoizedC = currMemoizedC
        self.memoizedD = currMemoizedD
        self.prevItem = currprevItem
        self.prevDetVal = currprevDetVal
        return result

    def evaluate_with_memoization(self, X):
        return self.prevDetVal

    def marginal_gain(self, X, item):
        currMemoizedC = self.memoizedC.copy()
        currMemoizedD = self.memoizedD.copy()
        currprevItem = self.prevItem
        currprevDetVal = self.prevDetVal
        self.set_memoization(X)
        result = self.marginal_gain_with_memoization(X, item)
        self.memoizedC = currMemoizedC
        self.memoizedD = currMemoizedD
        self.prevItem = currprevItem
        self.prevDetVal = currprevDetVal
        return result

    def marginal_gain_with_memoization(self, X, item, enableChecks=True):
        effectiveX = X.intersection(self.effective_ground_set) if self.partial else X
        gain = 0

        if enableChecks and item in effectiveX:
            return 0

        if self.partial and item not in self.effective_ground_set:
            return 0

        itemIndex = self.originalToPartialIndexMap[item] if self.partial else item

        if self.mode == "dense":
            if len(effectiveX) == 0:
                gain = math.log(self.memoizedD[itemIndex] * self.memoizedD[itemIndex])
            elif len(effectiveX) == 1:
                prevItemIndex = self.originalToPartialIndexMap[self.prevItem] if self.partial else self.prevItem
                e = self.dense_kernel[self.prevItem][item] / self.memoizedD[prevItemIndex]
                gain = math.log(math.fabs(self.memoizedD[itemIndex] * self.memoizedD[itemIndex] - e * e))
            else:
                prevItemIndex = self.originalToPartialIndexMap[self.prevItem] if self.partial else self.prevItem
                e = (self.dense_kernel[self.prevItem][item] -
                     self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[itemIndex])) / self.memoizedD[prevItemIndex]
                gain = math.log(math.fabs(self.memoizedD[itemIndex] * self.memoizedD[itemIndex] - e * e))
        elif self.mode == "sparse":
            if len(effectiveX) == 0:
                gain = math.log(math.fabs(self.memoizedD[itemIndex] * self.memoizedD[itemIndex]))
            elif len(effectiveX) == 1:
                prevItemIndex = self.originalToPartialIndexMap[self.prevItem] if self.partial else self.prevItem
                e = self.sparseKernel.get_val(self.prevItem, item) / self.memoizedD[prevItemIndex]
                gain = math.log(math.fabs(self.memoizedD[itemIndex] * self.memoizedD[itemIndex] - e * e))
            else:
                prevItemIndex = self.originalToPartialIndexMap[self.prevItem] if self.partial else self.prevItem
                e = (self.sparseKernel.get_val(self.prevItem, item) -
                     self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[itemIndex])) / self.memoizedD[prevItemIndex]
                gain = math.log(math.fabs(self.memoizedD[itemIndex] * self.memoizedD[itemIndex] - e * e))
        else:
            raise ValueError("Only dense and sparse mode supported")

        return gain

    def update_memoization(self, X, item):
        effectiveX = X.intersection(self.effective_ground_set) if self.partial else X

        if item in effectiveX:
            return

        if item not in self.effective_ground_set:
            return

        self.prevDetVal += self.marginal_gain_with_memoization(X, item)

        if len(effectiveX) == 0:
            pass
        else:
            prevItemIndex = self.originalToPartialIndexMap[self.prevItem] if self.partial else self.prevItem
            prevDValue = self.memoizedD[prevItemIndex]

            for i in self.effectiveGroundSet:
                iIndex = self.originalToPartialIndexMap[i] if self.partial else i

                if i in effectiveX:
                    continue

                e = 0
                if len(effectiveX) == 1:
                    e = self.dense_kernel[self.prevItem][i] / prevDValue
                    self.memoizedC[iIndex].append(e)
                else:
                    e = (self.dense_kernel[self.prevItem][i] -
                         self.dot_product(self.memoizedC[prevItemIndex], self.memoizedC[iIndex])) / prevDValue
                    self.memoizedC[iIndex].append(e)

                self.memoizedD[iIndex] = math.sqrt(math.fabs(self.memoizedD[iIndex] * self.memoizedD[iIndex] - e * e))

        self.prevItem = item

    def get_effective_ground_set(self):
        return self.effective_ground_set

    def clear_memoization(self):
        self.memoizedC.clear()
        self.memoizedC = defaultdict(list)
        self.prevDetVal = 0
        self.prevItem = -1

        if self.mode == "dense":
            if self.partial:
                for it in self.effective_ground_set:
                    index = self.originalTo
