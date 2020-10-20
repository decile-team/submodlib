# clustered.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>

from .setFunction import SetFunction

class ClusteredFunction(SetFunction):
	"""Implementation of the Clustered function

	Given a function and a clustering, clustered function internally creates a mixture of function on each cluster. It is defined as
	
	.. math::
	        f(X) = \\sum_i f_{C_i}(X \\cap C_i)
	
	.. note::
	        When the clusters are labels, this becomes supervised subset selection

    Parameters
	----------

	n : int
	    Number of elements in the ground set
	
	f : SetFunction
	    The particular instantiated set function whose clustered version is desired
	
	clusters : list
	    List of clusters each containing set of items in the ground set belonging to that cluster	
	
	
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