# saturatedCoverage.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class SaturatedCoverageFunction(SetFunction):
	"""Implementation of the SaturatedCoverage submodular function
	
	Saturated Coverage is similar to Facility Location except for the fact that for every category, instead of taking a single representative, it allows for taking potentially multiple representatives. It is defined as

	.. math::
	        f_{satc}(X) = \\sum_{v \\in V} min\\{m_v(X), c\\}
	
	where :math:`m_v(X) = \\sum_{x \\in X} sim(v, x)` measures the relevance of set :math:`X` to item :math:`v \\in V` and :math:`c` is a saturation hyper parameter that controls the level of coverage for each item :math:`v` by the set :math:`X`.

	Saturated Coverage is monotone submodular and is a special instance of the generalized set-cover. 

	.. note::
	        Saturated Coverage function can also be seen as a Concave Over odular function with kernel.
	
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