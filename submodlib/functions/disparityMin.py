# disparityMin.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class DisparityMinFunction(SetFunction):
	"""Implementation of the Disparity-Min function
	
	Diversity based functions attempt to obtain a diverse set of keypoints. The goal is to have minimum similarity across elements in the chosen subset by maximizing minimum pairwise distance between elements. There is a subtle difference between the notion of diversity and the notion of representativeness. While diversity *only* looks at the elements in the chosen subset, representativeness also worries about their similarity with the remaining elements in the superset. Denote :math:`d_{ij}` as a distance measure between images/data points :math:`i` and :math:`j`. Disparity-Min for a subset :math:`X` is defined as: 
	
	.. math::
	        f(X) = \\min_{i, j \\in X, i \\neq j} d_{ij}

	This function is not submodular, but can be efficiently optimized via a greedy algorithm :cite:`dasgupta2013summarization`. It is easy to see that maximizing this function involves obtaining a subset with maximal minimum pairwise distance, thereby ensuring a diverse subset of datapoints (snippets or keyframes) in the summary.

	Parameters
	----------

	n : int
	    Number of elements in the ground set
	
	dijs : list, optional
	    Distance matrix to be used for getting :math:`d_{ij}` entries as defined above. When None, it is computed based on the following additional parameters

	data : list, optional
	    Data matrix which will be used for computing the distance matrix

	metric : str, optional
	    Distance metric to be used for computing the distance matrix
	
	n_neighbors : int, optional
	    While constructing distance matrix, number of nearest neighbors whose distance values will be kept resulting in a sparse distance matrix for computation speed up (at the cost of accuracy)

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