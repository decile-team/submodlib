# disparitySum.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class DisparitySumFunction(SetFunction):
	"""Implementation of the Disparity-Sum function

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