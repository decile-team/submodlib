# probabilisticSetCover.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class ProbabilisticSetCoverFunction(SetFunction):
	"""Implementation of the Probabilistic Set Cover submodular function.
	
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

	def __init__(self, n, n_concepts, probs, weights):
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