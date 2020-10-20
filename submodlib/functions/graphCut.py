# graphCut.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class GraphCutFunction(SetFunction):
	"""Implementation of the Graph-Cut submodular function
	
	Graph-Cut models representation and is defined as

	.. math::
	        f_{gc}(X) = \\sum_{i \\in V, j \\in X} sim(i, j) - \\lambda \\sum_{i, j \\in X} sim(i, j)
	
	When :math:`\\lambda` becomes large, it also tries to model diversity in the subset. :math:`\\lambda` governs the tradeoff between representation and diversity.

    .. note::
	        For :math:`\\lambda < 0.5` Graph-Cut is monotone submodular. For :math:`\lambda > 0.5` it is non-monotone submodular.
	
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