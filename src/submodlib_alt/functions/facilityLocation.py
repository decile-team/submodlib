# facilityLocation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class FacilityLocationFunction(SetFunction):
	"""Implementation of the Facility-Location submodular function.
	
	Facility-Location function :cite:`mirchandani1990discrete` attempts to model representation, as in it tries to find a representative subset of items, akin to centroids and medoids in clustering. The Facility-Location function is closely related to k-medoid clustering. While diversity *only* looks at the elements in the chosen subset, representativeness also worries about their similarity with the remaining elements in the superset. Denote :math:`s_{ij}` as the similarity between images/datapoints :math:`i` and :math:`j`. We can then define 

	.. math::
			f(X) = \\sum_{i \\in V} \\max_{j \\in X} s_{ij} 
	
	For each image :math:`i` in the ground set :math:`V`, we compute the representative from subset :math:`X` which is closest to :math:`i` and add these similarities for all images. 

	Facility-Location is monotone submodular.
	
	.. note:: 
		This function requires computing a :math:`\\mathcal{O}(n^2)` similarity function. However, as shown in :cite:`wei2014fast`, we can approximate this with a nearest neighbor graph, which will require much less storage, and also can run much faster for large ground set sizes.

	Parameters
	----------

	n : int
		Number of elements in the ground set
	
	sijs : list, optional
		Similarity matrix to be used for getting :math:`s_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the similarity matrix

	metric : str, optional
		Similarity metric to be used for computing the similarity matrix
	
	n_neighbors : int, optional
		While constructing similarity matrix, number of nearest neighbors whose similarity values will be kept resulting in a sparse similarity matrix for computation speed up (at the cost of accuracy)

	"""

	def __init__(self, n, sijs, data, metric, n_neighbors):
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