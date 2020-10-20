# setCover.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class SetCoverFunction(SetFunction):
	"""Implementation of the Set-Cover submodular function
	
	For a subset :math:`X` being scored, the set cover is defined as 
	
	.. math::
	        f_{sc}(X) = \\sum_{u \\in U} min\\{m_u(X), 1\\}
	
	:math:`u` is a concept belonging to a set of all concepts :math:`U`, :math:`m_u(X) = \\sum_{x \\in X} w_{xu}` and :math:`w_{xu}` is the weight of coverage of concept :math:`u` by element :math:`x`. This functions models the coverage aspect of a candidate summary (subset) is monotone submodular.

	.. math::
	        f(A) = w(\\cup_{a \\in A} \\gamma(a)) = w(\\gamma(A))
	
	where :math:`w` is a weight vector in :math:`\\mathbb{R}^{\\gamma(\\Omega)}`. Intuitively, each element in :math:`\\Omega` *covers* a set of elements from the concept set :math:`U` and hence :math:`w(\\gamma(A))` is total weight of concepts covered by elements in :math:`A`. Note that :math:`\\gamma(A \\cup B) = \\gamma(A) \\cup \\gamma(B)` and hence :math:`f(A \\cup B) = w(\\gamma(A \\cup B)) = w(\\gamma(A) \\cup \\gamma(B))`
		
	Alternatively we can also view the function as follows. With :math:`U` being the set of all concepts (namely :math:`U = \\gamma(\\Omega)`) and :math:`c_u(i)` denoting whether the concept :math:`u \\in U` is covered by the element :math:`i \\in \\Omega` i.e :math:`c_u(i) = 1` if :math:`u \\in \\gamma(\\{i\\})` and is zero otherwise. We then define :math:`c_u(A) = \\sum_{a\\in A} c_u(a)` as the count of concept :math:`u` in set :math:`A`, and the weighted set cover then is
		
	.. math::
	        f(A) = \\sum_{u \\in U} w_u \\min(c_u(A), 1)

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