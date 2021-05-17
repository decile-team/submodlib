# probabilisticSetCoverMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
from submodlib_cpp import ProbabilisticSetCoverMutualInformation

class ProbabilisticSetCoverMutualInformationFunction(SetFunction):
	"""Implementation of the Probabilistic Set Cover Conditional Gain submodular function.
	
	This variant of the set cover function is defined as 
	
	.. math::
			f_{psc}(X) = \\sum_{u \\in U} (1 - \\prod_{x \\in X} (1 - p_{xu}))
	
	where :math:`p_{xu}` is the probability with which concept :math:`u` is covered by element :math:`x`. Similar to the set cover function, this function models the coverage aspect of the candidate summary (subset), viewed stochastically and is also monotone submodular.

	The probabilistic set cover function is 
	
	.. math::
			f(A) = \\sum_{i \\in U} w_i(1 - P_i(A))
	
	where :math:`U` is the set of concepts, and :math:`P_i(A) = \\prod_{j \\in A} (1 - p_{ij})`, i.e. :math:`P_i(A)` is the probability that :math:`A` *doesn't* cover concept :math:`i`. Intuitively, PSC is a soft version of the SC, which allows for probability of covering concepts, instead of a binary yes/no, as is the case with SC.

	Parameters
	----------
	n : int
		Number of elements in the ground set
	n_concepts : int
		Number of concepts
	probs : list
		List of probability vectors for each data point / image, each probability vector containing the probabilities with which that data point / image covers each concept
	weights : list
		Weight :math:`w_i` of each concept
	
	"""

	def __init__(self, n, num_concepts, probs, query_concepts, concept_weights=None):
		self.n = n
		self.num_concepts = num_concepts
		self.probs = probs
		self.query_concepts = query_concepts
		self.concept_weights = concept_weights
		self.cpp_obj = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.n != len(self.probs):
			raise Exception("ERROR: Mismtach between n and len(probs)")

		if self.num_concepts != len(self.probs[0]):
			raise Exception("ERROR: Mismtach between num_concepts and len(probs[0])")
		
		if (type(self.concept_weights) != type(None)):
			if self.num_concepts != len(self.concept_weights):
			    raise Exception("ERROR: Mismtach between num_conepts and len(concept_weights)")
		else:
			self.concept_weights = [1] * self.num_concepts

		self.cpp_obj = ProbabilisticSetCoverMutualInformation(self.n, self.num_concepts, self.probs, self.concept_weights, self.query_concepts)

		self.effective_ground = set(range(n))

	def evaluate(self, X):
		"""Computes the Probabilistic Set Cover Conditional Gain score of a set

		Parameters
		----------
		X : set
			The set whose Probabilistic Set Cover Conditional Gain score needs to be computed
		
		Returns
		-------
		float
			The Probabilistic Set Cover Conditional Gain function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False):
		"""Find the optimal subset with maximum Probabilistic Set Cover Conditional Gain score for a given budget

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
		"""Find the marginal gain in Probabilistic Set Cover Conditional Gain score when a single item (element) is added to a set (X)

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
		"""Efficiently find the marginal gain in Probabilistic Set Cover Conditional Gain score when a single item (element) is added to a set (X) assuming that memoized statistics for it are already computed

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
		"""Efficiently compute the Probabilistic Set Cover Conditional Gain score of a set assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			The set whose Probabilistic Set Cover Conditional Gain score needs to be computed
		
		Returns
		-------
		float
			The Probabilistic Set Cover Conditional Gain function evaluation on the given set

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
		"""Get the effective ground set of this Probabilistic Set Cover Conditional Gain object. This is equal to the ground set when instantiated with partial=False and is equal to the ground_sub when instantiated with partial=True

		"""
		return self.effective_ground