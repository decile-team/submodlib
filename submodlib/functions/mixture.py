# mixture.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>

from .setFunction import SetFunction

class MixtureFunction(SetFunction):
	"""Implementation of the Mixture function.
	
	Given a set of different function instantiations :math:`f_1, f_2, f_3, ... f_n` and their respective weights :math:`w_1, w_2, w_3, ... w_n`, the (weighted) mixture function is defined as 

	.. math::
			f(X) = \\sum_i w_i f_i

	Parameters
	----------
	functions : list
		List of instantiated functions in the mixture
	weights : list
		Corresponding weights
	
	"""

	def __init__(self, functions, weights):
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