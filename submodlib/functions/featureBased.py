# featureBased.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class FeatureBasedFunction(SetFunction):
	"""Implementation of the Feature-Based function

	Feature based functions are essentially sums of concave over modular functions defined as
	
	.. math::
	        f(X) = \\sum_{f \\in F} w_f g(m_f(X))
	
	where :math:`g` is a concave function, :math:`{m_f}_{f \\in F}` are a set of feature scores, and :math:`f \\in F` are features.

	
	
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