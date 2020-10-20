# probabilisticSetCover.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class ProbabilisticSetCoverFunction(SetFunction):
	"""Implementation of the Probabilistic Set Cover submodular function
	
	This variant of the set cover function is defined as 
	
	.. math::
	        f_{psc}(X) = \\sum_{u \\in U} (1 - \\prod_{x \\in X} (1 - p_{xu}))
	
	where :math:`p_{xu}` is the probability with which concept :math:`u` is covered by element :math:`x`. Similar to the set cover function, this function models the coverage aspect of the candidate summary (subset), viewed stochastically and is also monotone submodular.

    The probabilistic set cover function is 
	
    .. math::
	        f(A) = \\sum_{i \\in U} w_i(1 - P_i(A))
	
	where :math:`U` is the set of concepts, and :math:`P_i(A) = \\prod_{j \\in A} (1 - p_{ij})`, i.e. :math:`P_i(A)` is the probability that :math:`A` *doesn't* cover concept :math:`i`. Intuitively, PSC is a soft version of the SC, which allows for probability of covering concepts, instead of a binary yes/no, as is the case with SC.
	
	"""

	def __init__():
		pass

	def evaluate():
		"""Computes the score of a set

		"""

		pass

	def maximize():
		"""Find the optimal subset with maximum score

		"""

		pass
	
	def marginalGain():
		"""Find the marginal gain of adding an item to a set

		"""

		pass