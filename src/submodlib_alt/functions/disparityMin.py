# disparityMin.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class DisparityMinFunction(SetFunction):
	"""Implementation of the Disparity-Min function.
	
	Diversity based functions attempt to obtain a diverse set of keypoints. The goal is to have minimum similarity across elements in the chosen subset by maximizing minimum pairwise distance between elements. There is a subtle difference between the notion of diversity and the notion of representativeness. While diversity *only* looks at the elements in the chosen subset, representativeness also worries about their similarity with the remaining elements in the superset. Denote :math:`d_{ij}` as a distance measure between images/data points :math:`i` and :math:`j`. Disparity-Min for a subset :math:`X` is defined as: 
	
	.. math::
			f(X) = \\min_{i, j \\in X, i \\neq j} d_{ij}

	It is easy to see that maximizing this function involves obtaining a subset with maximal minimum pairwise distance, thereby ensuring a diverse subset of datapoints (snippets or keyframes) in the summary.

	.. note::
			This function is not submodular, but can be efficiently optimized via a greedy algorithm :cite:`dasgupta2013summarization`.

	Parameters
	----------

	n : int
		Number of elements in the ground set
	
	dijs : list, optional
		Distance matrix to be used for getting :math:`d_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the distance matrix

	metric : str, optional
		Distance metric to be used for computing the distance matrix
	
	n_neighbors : int, optional
		While constructing distance matrix, number of nearest neighbors whose distance values will be kept resulting in a sparse distance matrix for computation speed up (at the cost of accuracy)

	"""

	def __init__(self, n, dijs, data, metric, n_neighbors):
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