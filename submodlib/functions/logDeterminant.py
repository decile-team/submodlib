# logDeterminant.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class LogDeterminantFunction(SetFunction):
	"""Implementation of the Log-Determinant function

	Given a positive semi-definite kernel matrix :math:`L`, denote :math:`L_A` as the subset of rows and columns indexed by the set :math:`A`. The log-determinant function is 
	
	.. math::
	        f(A) = \\log\\det(L_A)
	
	The log-det function models diversity, and is closely related to a determinantal point process :cite:`kulesza2012determinantal`.
	
	Determinantal Point Processes (DPP) are an example of functions that model diversity. DPPs are defined as
	
	.. math::
	        p(X) = \\mbox{Det}(S_X)
	
	where :math:`S` is a similarity kernel matrix, and :math:`S_X` denotes the rows and columns instantiated with elements in :math:`X`. It turns out that :math:`f(X) = \\log p(X)` is submodular, and hence can be efficiently optimized via the greedy algorithm.	

	.. note::
	        DPP requires computing the determinant and is :math:`\\mathcal{O}(n^3)` where :math:`n` is the size of the ground set.
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