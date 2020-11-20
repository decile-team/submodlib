# setFunction.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>

class SetFunction():
	"""Generic base class for all set functions
	
	"""

	def __init__(self, n, f, clusters):
		pass

	def evaluate(X):
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

		pass

	def maximize(budget, optimizer):
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
	
	def marginalGain(X, element):
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

		pass