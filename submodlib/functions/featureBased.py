# featureBased.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
import numpy as np
from .setFunction import SetFunction
import submodlib_cpp as subcp
from submodlib_cpp import FeatureBased
from sklearn.preprocessing import MinMaxScaler

class FeatureBasedFunction(SetFunction):
	"""Implementation of the Feature-Based function.

	Feature based functions are essentially sums of concave over modular functions defined as
	
	.. math::
			f(X) = \\sum_{f \\in F} w_f g(m_f(X))
	
	where :math:`g` is a concave function, :math:`{m_f}_{f \\in F}` are a set of feature scores, and :math:`f \\in F` are features.

	Parameters
	----------
	n : int
		Number of elements in the ground set

	type : str
		Concave function to be used - sqrt, log, min, pow
	
	features : list
		Feature vectors for the elements in the ground set
	
	weights : list
		Weights of features
	
	thresh : float
		Threshold to be used for min function
	
	pow : int
		Exponent to be used for power function
	
	"""

	def __init__(self, n, features, numFeatures, sparse, featureWeights=None, mode=FeatureBased.logarithmic):
		self.n = n
		self.mode = mode
		self.features = features
		self.numFeatures = numFeatures
		self.featureWeights = featureWeights
		self.cpp_obj = None
		self.cpp_features = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.mode not in [FeatureBased.squareRoot, FeatureBased.inverse, FeatureBased.logarithmic]:
			raise Exception("ERROR: Incorrect mode. Must be one of 'squareRoot', 'inverse' or 'logarithmic'")

		if n != len(features):
			raise Exception("ERROR: Mismtach between n and len(features)")
		
		if (type(featureWeights) != type(None)):
			if numFeatures != len(featureWeights):
			    raise Exception("ERROR: Mismtach between numFeatures and len(featureWeights)")
		else:
			self.featureWeights = [1] * numFeatures

		#print("Features before normalization: ", features)
		
		#min-max normalize the features so that they are between 0 and 1
		featuresArray = np.array(features)
		norm = MinMaxScaler().fit(featuresArray)
		normalizedFeatures = norm.transform(featuresArray)
		features = normalizedFeatures.tolist()
		#print("Features after normalization: ", features)
		#convert the features into sparse representation (list of tuples) if not already
		self.cpp_features = []
		if not sparse:
			for i in range(len(features)):
				featureVec = []
				for j in range(len(features[i])):
					if (features[i][j] != 0):
						featureVec.append((j, features[i][j]))
				self.cpp_features.append(featureVec)
		else:
			self.cpp_features = features

		#print("Sparse representation:", self.cpp_features)
			
		self.cpp_obj = FeatureBased(self.n, self.mode, self.cpp_features, self.numFeatures, self.featureWeights)

		self.effective_ground = set(range(n))

	def evaluate(self, X):
		"""Computes the Feature Based score of a set

		Parameters
		----------
		X : set
			The set whose Feature Based score needs to be computed
		
		Returns
		-------
		float
			The Feature Based function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False):
		"""Find the optimal subset with maximum Feature Based score for a given budget

		Parameters
		----------
		budget : int
			Desired size of the optimal set
		optimizer : string
			The optimizer that should be used to compute the optimal set. Can be 'NaiveGreedy', 'LazyGreedy', 'LazierThanLazyGreedy'
		stopIfZeroGain : bool
			Set to True if budget should be filled with items adding zero gain. If False, size of optimal set can be potentially less than the budget
		stopIfNegativeGain : bool
			Set to True if maximization should terminate as soon as the best gain in an iteration is negative. This can potentially lead to optimal set of size less than the budget
		verbose : bool
			Set to True to trace the execution of the maximization algorithm

		Returns
		-------
		set
			The optimal set of size budget

		"""

		if budget >= len(self.effective_ground):
			raise Exception("Budget must be less than effective ground set size")
		return self.cpp_obj.maximize(optimizer, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose)
	
	def marginalGain(self, X, element):
		"""Find the marginal gain in Feature Based score when a single item (element) is added to a set (X)

		Parameters
		----------
		X : set
			Set on which the marginal gain of adding an element has to be calculated. It must be a subset of the effective ground set.
		element : int
			Element for which the marginal gain is to be calculated. It must be from the effective ground set.

		Returns
		-------
		float
			Marginal gain of adding element to X

		"""

		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if type(element)!=int:
			raise Exception("ERROR: element should be an int")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X is not a subset of effective ground set")

		if element not in self.effective_ground:
			raise Exception("Error: element must be in the effective ground set")

		if element in X:
			return 0

		return self.cpp_obj.marginalGain(X, element)

	def marginalGainWithMemoization(self, X, element):
		"""Efficiently find the marginal gain in Feature Based score when a single item (element) is added to a set (X) assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			Set on which the marginal gain of adding an element has to be calculated. It must be a subset of the effective ground set and its memoized statistics should have already been computed
		element : int
			Element for which the marginal gain is to be calculated. It must be from the effective ground set.

		Returns
		-------
		float
			Marginal gain of adding element to X

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if type(element)!=int:
			raise Exception("ERROR: element should be an int")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X is not a subset of effective ground set")

		if element not in self.effective_ground:
			raise Exception("Error: element must be in the effective ground set")

		if element in X:
			return 0

		return self.cpp_obj.marginalGainWithMemoization(X, element)

	def evaluateWithMemoization(self, X):
		"""Efficiently compute the Feature Based score of a set assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			The set whose Feature Based score needs to be computed
		
		Returns
		-------
		float
			The Feature Based function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0

		return self.cpp_obj.evaluateWithMemoization(X)

	def updateMemoization(self, X, element):
		"""Update the memoized statistics of X due to adding element to X. Assumes that memoized statistics are already computed for X

		Parameters
		----------
		X : set
			Set whose memoized statistics must already be computed and to which the element needs to be added for the sake of updating the memoized statistics
		element : int
			Element that is being added to X leading to update of memoized statistics. It must be from effective ground set.

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if type(element)!=int:
			raise Exception("ERROR: element should be an int")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X is not a subset of effective ground set")

		if element not in self.effective_ground:
			raise Exception("Error: element must be in the effective ground set")

		if element in X:
			return

		self.cpp_obj.updateMemoization(X, element)
	
	def clearMemoization(self):
		"""Clear the computed memoized statistics, if any

		"""
		self.cpp_obj.clearMemoization()
	
	def setMemoization(self, X):
		"""Compute and store the memoized statistics for subset X 

		Parameters
		----------
		X : set
			The set for which memoized statistics need to be computed and set
		
		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		self.cpp_obj.setMemoization(X)
	
	def getEffectiveGroundSet(self):
		"""Get the effective ground set of this Feature Based object. This is equal to the ground set when instantiated with partial=False and is equal to the ground_sub when instantiated with partial=True

		"""
		return self.effective_ground