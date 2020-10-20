# facilityLocation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class FacilityLocationFunction(SetFunction):
	"""Implementation of the Facility-Location submodular function
	
	Facility-Location function :cite:`mirchandani1990discrete` attempts to model representation, as in it tries to find a representative subset of items, akin to centroids and medoids in clustering. The Facility-Location function is closely related to k-medoid clustering. While diversity *only* looks at the elements in the chosen subset, representativeness also worries about their similarity with the remaining elements in the superset. Denote :math:`s_{ij}` as the similarity between images/datapoints :math:`i` and :math:`j`. We can then define 

	.. math::
			f(X) = \\sum_{i \\in V} \\max_{j \\in X} s_{ij} 
	
	For each image :math:`i` in the ground set :math:`V`, we compute the representative from subset :math:`X` which is closest to :math:`i` and add these similarities for all images. 

	Facility-Location is monotone submodular.
	
	.. note:: 
		This function requires computing a :math:`\\mathcal{O}(n^2)` similarity function. However, as shown in :cite:`wei2014fast`, we can approximate this with a nearest neighbor graph, which will require much less storage, and also can run much faster for large ground set sizes.
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