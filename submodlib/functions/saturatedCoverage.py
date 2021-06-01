# saturatedCoverage.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction

class SaturatedCoverageFunction(SetFunction):
	"""Implementation of the SaturatedCoverage submodular function.
	
	Saturated Coverage is similar to Facility Location except for the fact that for every category, instead of taking a single representative, it allows for taking potentially multiple representatives. It is defined as

	.. math::
			f_{satc}(X) = \\sum_{v \\in V} \\min{m_v(X), c}
	
	where :math:`m_v(X) = \\sum_{x \\in X} sim(v, x)` measures the relevance of set :math:`X` to item :math:`v \\in V` and :math:`c` is a saturation hyper parameter that controls the level of coverage for each item :math:`v` by the set :math:`X`.

	.. math::
			f(X) = \\sum_{i \\in V} \\min{\\sum_{j \\in X} s_{ij}, \\alpha \\sum_{j \\in V} s_{ij}}

	Saturated Coverage is monotone submodular and is a special instance of the generalized set-cover. 

	.. note::
			Saturated Coverage function can also be seen as a Concave Over Modular function with kernel.

	Parameters
	----------
	n : int
		Number of elements in the ground set
	
	alpha : float
		Constant as defined above
	
	sijs : list, optional
		Similarity matrix to be used for getting :math:`s_{ij}` entries as defined above. When not provided, it is computed based on the following additional parameters

	data : list, optional
		Data matrix which will be used for computing the similarity matrix

	metric : str, optional
		Similarity metric to be used for computing the similarity matrix
	
	n_neighbors : int, optional
		While constructing similarity matrix, number of nearest neighbors whose similarity values will be kept resulting in a sparse similarity matrix for computation speed up (at the cost of accuracy)

	"""

	def __init__(self, n, alpha, sijs, data, metric, n_neighbors):
		pass

	