# clustered.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>

from .setFunction import SetFunction

class ClusteredFunction(SetFunction):
	"""Implementation of the Clustered function.

	Given a function and a clustering, clustered function internally creates a mixture of function on each cluster. It is defined as
	
	.. math::
			f(X) = \\sum_i f_{C_i}(X \\cap C_i)
	
	.. note::
			When the clusters are labels, this becomes supervised subset selection.

	Parameters
	----------
	n : int
		Number of elements in the ground set
	f : SetFunction
		The particular instantiated set function whose clustered version is desired
	clusters : list
		List of clusters each containing set of items in the ground set belonging to that cluster	
	
	"""

	def __init__(self, n, f, clusters):
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

		pass

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

		pass