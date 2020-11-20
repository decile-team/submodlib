# graphCut.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class GraphCutFunction(SetFunction):
	"""Implementation of the Graph-Cut submodular function.
	
	Graph-Cut models representation and is defined as

	.. math::
			f_{gc}(X) = \\sum_{i \\in V, j \\in X} s_{ij} - \\lambda \\sum_{i, j \\in X} s_{ij}
	
	When :math:`\\lambda` becomes large, graph cut function also tries to model diversity in the subset. :math:`\\lambda` governs the tradeoff between representation and diversity.

	.. note::
			For :math:`\\lambda < 0.5` Graph-Cut is monotone submodular. For :math:`\lambda > 0.5` it is non-monotone submodular.

	Parameters
	----------
	n : int
		Number of elements in the ground set

	lam : float
		Trade-off between representation and diversity as defined by :math:`\\lambda` in the above definition
	
	sijs : list, optional
		Similarity matrix to be used for getting :math:`s_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the similarity matrix

	metric : str, optional
		Similarity metric to be used for computing the similarity matrix
	
	n_neighbors : int, optional
		While constructing similarity matrix, number of nearest neighbors whose similarity values will be kept resulting in a sparse similarity matrix for computation speed up (at the cost of accuracy)
	
	"""

	def __init__(self, n, lam, sijs, data, metric, n_neighbors):
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