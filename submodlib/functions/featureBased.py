# featureBased.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

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

	def __init__(self, n, type, features, weights, thresh, pow):
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