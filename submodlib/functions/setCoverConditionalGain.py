# setCoverConditionalGain.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
from submodlib_cpp import SetCoverConditionalGain

class SetCoverConditionalGainFunction(SetFunction):
	"""Implementation of the Set-Cover submodular function.
	
	For a subset :math:`X` being scored, the set cover is defined as 
	
	.. math::
			f_{sc}(X) = \\sum_{u \\in U} min\\{m_u(X), 1\\}
	
	:math:`u` is a concept belonging to a set of all concepts :math:`U`, :math:`m_u(X) = \\sum_{x \\in X} w_{xu}` and :math:`w_{xu}` is the weight of coverage of concept :math:`u` by element :math:`x`. This functions models the coverage aspect of a candidate summary (subset) is monotone submodular.

	.. math::
			f(A) = w(\\cup_{a \\in A} \\gamma(a)) = w(\\gamma(A))
	
	where :math:`w` is a weight vector in :math:`\\mathbb{R}^{\\gamma(\\Omega)}`. Intuitively, each element in :math:`\\Omega` *covers* a set of elements from the concept set :math:`U` and hence :math:`w(\\gamma(A))` is total weight of concepts covered by elements in :math:`A`. Note that :math:`\\gamma(A \\cup B) = \\gamma(A) \\cup \\gamma(B)` and hence :math:`f(A \\cup B) = w(\\gamma(A \\cup B)) = w(\\gamma(A) \\cup \\gamma(B))`
		
	Alternatively we can also view the function as follows. With :math:`U` being the set of all concepts (namely :math:`U = \\gamma(\\Omega)`) and :math:`c_u(i)` denoting whether the concept :math:`u \\in U` is covered by the element :math:`i \\in \\Omega` i.e :math:`c_u(i) = 1` if :math:`u \\in \\gamma(\\{i\\})` and is zero otherwise. We then define :math:`c_u(A) = \\sum_{a\\in A} c_u(a)` as the count of concept :math:`u` in set :math:`A`, and the weighted set cover then is
		
	.. math::
			f(A) = \\sum_{u \\in U} w_u \\min(c_u(A), 1)
	
	Parameters
	----------
	n : int
		Number of elements in the ground set

	cover_set : list
		List of sets. Each set is the set of concepts covered by the corresponding data point / image

	weights : float
		Weights of concepts

	"""

	def __init__(self, n, cover_set, num_concepts, private_concepts, concept_weights=None):
		self.n = n
		self.cover_set = cover_set
		self.num_concepts = num_concepts
		self.private_concepts = private_concepts
		self.concept_weights = concept_weights
		self.cpp_obj = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.n != len(self.cover_set):
			raise Exception("ERROR: Mismtach between n and len(cover_set)")
		
		if (type(self.concept_weights) != type(None)):
			if self.num_concepts != len(self.concept_weights):
			    raise Exception("ERROR: Mismtach between num_conepts and len(concept_weights)")
		else:
			self.concept_weights = [1] * self.num_concepts

		self.cpp_obj = SetCoverConditionalGain(self.n, self.cover_set, self.num_concepts, self.concept_weights, self.private_concepts)

		self.effective_ground = set(range(n))

	def evaluate(self, X):
		"""Computes the Set Cover score of a set

		Parameters
		----------
		X : set
			The set whose Set Cover score needs to be computed
		
		Returns
		-------
		float
			The Set Cover function evaluation on the given set

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False):
		"""Find the optimal subset with maximum Set Cover score for a given budget

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
		"""Find the marginal gain in Set Cover score when a single item (element) is added to a set (X)

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
		"""Efficiently find the marginal gain in Set Cover score when a single item (element) is added to a set (X) assuming that memoized statistics for it are already computed

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
		"""Efficiently compute the Set Cover score of a set assuming that memoized statistics for it are already computed

		Parameters
		----------
		X : set
			The set whose Set Cover score needs to be computed
		
		Returns
		-------
		float
			The Set Cover function evaluation on the given set

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
		"""Get the effective ground set of this Set Cover object. This is equal to the ground set when instantiated with partial=False and is equal to the ground_sub when instantiated with partial=True

		"""
		return self.effective_ground
