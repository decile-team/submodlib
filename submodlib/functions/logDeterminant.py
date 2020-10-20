# logDeterminant.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class LogDeterminantFunction(SetFunction):
	"""Implementation of the Log-Determinant function.

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

	Parameters
	----------
	n : int
		Number of elements in the ground set

	lam : float
		Addition to :math:`s_{ii} (1)` so that :math:`\\log` doesn't become 0 
	
	L : list, optional
		Similarity matrix L. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the similarity matrix

	metric : str, optional
		Similarity metric to be used for computing the similarity matrix
	
	"""

	def __init__(self, n, lam, L, data, metric):
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