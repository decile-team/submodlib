# setFunction.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>

class SetFunction():
	"""Generic base class for all set functions
	
	"""

	def __init__(self, n, f, clusters):
		pass

	def evaluate(self, X):
		"""Computes the score of a set as per the above math.

		Parameters
		----------
		X : set
			The set whose score needs to be computed. Must be a subset of effective ground set. 
		
		Returns
		-------
		float
			The evaluation score of the given set.

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0
		return self.cpp_obj.evaluate(X)

	def maximize(self, budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, epsilon = 0.1, verbose=False, show_progress=True, costs=None, costSensitiveGreedy=False):
		"""Compute the optimal subset with maximum score for the given *budget*.

		Parameters
		----------
		budget : int
			Desired size of the optimal set.
		optimizer : string
			The optimizer that should be used to compute the optimal set. Can be 'NaiveGreedy', 'StochasticGreedy', LazyGreedy' and 'LazierThanLazyGreedy'.
		stopIfZeroGain : bool
			Set to True if maximization should terminate as soon as gain of adding any other item becomes zero. When True, size of optimal set can thus be potentially less than the budget.
		stopIfNegativeGain : bool
			Set to True if maximization should terminate as soon as the best gain in an iteration is negative. When True, this can potentially lead to optimal set of size less than the budget.
		epsilon : float
			Used by :ref:`optimizers.stochastic-greedy` and :ref:`optimizers.lazier-than-lazy-greedy` to compute the size of the random set.
		verbose : bool
			Set to True to trace/debug the execution of the maximization algorithm.
		show_progress : bool
			Set to True to see progress a progress bar.
		costs : list, optional
			List containing cost of each element of the ground set. Cost contributes to the budget. When *costSensitiveGreedy* is set to True, the marginal gain is divided by the cost to identify the next best element to add in every iteration. Default is None which means all ground set elements have cost = 1. It is possible to specify *costs* and yet have *costSensitiveGreedy* set to False. This would correspond use regular marginal gains, but the budget gets filled as per the costs of selected items. 
		costSensitiveGreedy : bool, optional
			When set to True, the next best candidate in every iteration is decided based on their marginal gain divided by cost. When True, it is mandatory to provide *costs*. Defaults to False.

		Returns
		-------
		set
			The optimal set of size *budget*.

		"""

		if budget >= len(self.effective_ground):
			raise Exception("Budget must be less than effective ground set size")
		if type(costs) == type(None):
			return self.cpp_obj.maximize(optimizer, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, show_progress, [], costSensitiveGreedy)
		else:
			if len(costs) != self.n:
				raise Exception("Mismtach between length of costs and number of elements in the groundset")
			return self.cpp_obj.maximize(optimizer, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, show_progress, costs, costSensitiveGreedy)

		
	
	def marginalGain(self, X, element):
		"""Computes the marginal gain in score of this function when a single item (*element*) is added to a set (*X*).

		Parameters
		----------
		X : set
			Set on which the marginal gain of adding an element has to be calculated. It must be a subset of the effective ground set.
		element : int
			Element for which the marginal gain is to be calculated. It must be from the effective ground set.

		Returns
		-------
		float
			Marginal gain of adding *element* to X.

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
		"""Efficiently find the marginal gain in score when a single item (*element*) is added to a set (*X*) assuming that memoized statistics for X are already computed.

		Parameters
		----------
		X : set
			Set on which the marginal gain of adding an element has to be calculated. It must be a subset of the effective ground set and its memoized statistics should have already been computed.
		element : int
			Element for which the marginal gain is to be calculated. It must be from the effective ground set.

		Returns
		-------
		float
			Marginal gain of adding *element* to X.

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

		return self.cpp_obj.marginalGainWithMemoization(X, element, True)

	def evaluateWithMemoization(self, X):
		"""Efficiently compute the function evaluation of a set assuming that memoized statistics for it are already computed.

		Parameters
		----------
		X : set
			The set on which the function needs to be evaluated. It must be a subset of the effective ground set.
		
		Returns
		-------
		float
			The function evaluation score on the given set.

		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		if len(X) == 0:
			return 0

		return self.cpp_obj.evaluateWithMemoization(X)

	def updateMemoization(self, X, element):
		"""Update the memoized statistics of a set *X* due to adding an element to it. Assumes that memoized statistics are already computed for *X*. Note that the element is **not** added to the set and only the memoized statistics are updated. The actual insertion of *element* to *X* is the responsibility of the caller.

		Parameters
		----------
		X : set
			Set whose memoized statistics must already be computed and to which the element needs to be added for the sake of updating the memoized statistics.
		element : int
			Element that is being added to X leading to update of memoized statistics. It must be from the effective ground set.

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
		"""Clear the computed memoized statistics, if any.

		"""
		self.cpp_obj.clearMemoization()
	
	def setMemoization(self, X):
		"""Compute and store the memoized statistics for subset *X*.

		Parameters
		----------
		X : set
			The set for which memoized statistics need to be computed and set, overwriting any existing memoized statistics.
		
		"""
		if type(X)!=set:
			raise Exception("ERROR: X should be a set")

		if X.issubset(self.effective_ground)==False:
			raise Exception("ERROR: X should be a subset of effective ground set")

		self.cpp_obj.setMemoization(X)
	
	def getEffectiveGroundSet(self):
		"""Get the effective ground set of this object. 

		"""
		return self.cpp_obj.getEffectiveGroundSet()