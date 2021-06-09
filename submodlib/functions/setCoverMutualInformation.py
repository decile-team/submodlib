# setCoverMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
from submodlib_cpp import SetCoverMutualInformation

class SetCoverMutualInformationFunction(SetFunction):
	"""Implementation of the Set Cover Mutual Information (SCMI) function.

	Given a :ref:`functions.submodular-mutual-information` function, Set Cover Mutual Information function is its instantiation using a :class:`~submodlib.functions.setCover.SetCoverFunction`. Mathematically, it takes the following form:

	.. math::
			I_f(A; Q) = w(\\gamma(A) \\cap \\gamma(Q))
	
	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	cover_set : list
		List of sets. Each set is the set of concepts covered by the corresponding data point / image. Hence cover_set is of size n.
	
	num_concepts : int
		Number of concepts.
	
	query_concepts : set
		Set of query concepts. That is, the concepts which should be covered by the optimal subset.
	
	concept_weights : list
		Weight :math:`w_i` of each concept. Size must be same as num_concepts.
	
	"""

	def __init__(self, n, cover_set, num_concepts, query_concepts, concept_weights=None):
		self.n = n
		self.cover_set = cover_set
		self.num_concepts = num_concepts
		self.query_concepts = query_concepts
		self.concept_weights = concept_weights
		self.cpp_obj = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.n != len(self.cover_set):
			raise Exception("ERROR: Mismtach between n and len(cover_set)")
		
		if (type(self.concept_weights) != type(None)):
			if self.num_concepts != len(self.concept_weights):
			    raise Exception("ERROR: Mismtach between num_conepts and len(concept_weights)")
		else:
			self.concept_weights = [1] * self.num_concepts

		self.cpp_obj = SetCoverMutualInformation(self.n, self.cover_set, self.num_concepts, self.concept_weights, self.query_concepts)

		self.effective_ground = set(range(n))

	