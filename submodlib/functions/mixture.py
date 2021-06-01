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

	