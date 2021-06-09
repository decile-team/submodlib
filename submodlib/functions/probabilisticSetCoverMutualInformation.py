# probabilisticSetCoverMutualInformation.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
from submodlib_cpp import ProbabilisticSetCoverMutualInformation

class ProbabilisticSetCoverMutualInformationFunction(SetFunction):
	"""Implementation of the Probabilistic Set Cover Mutual Information (PSCMI) function.

	Given a :ref:`functions.submodular-mutual-information` function, Probabilistic Set Cover Mutual Information function is its instantiation using a :class:`~submodlib.functions.probabilisticSetCover.ProbabilisticSetCoverFunction`. Mathematically, it takes the following form:

	.. math::
			I_f(A; Q) = \\sum\\limits_{u \\in \\mathcal{U}} w_u \\bar{P_u(A)} \\bar{P_u}(Q)
	
	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	num_concepts : int
		Number of concepts.
	
	probs : list
		List of probability vectors for each data point / image, each probability vector containing the probabilities with which that data point / image covers each concept. Hence each list is num_concepts dimensional and probs contains n such lists.
	
	query_concepts : set
		Set of query concepts. That is, the concepts which should be covered by the optimal subset.
	
	concept_weights : list
		Weight :math:`w_i` of each concept. Size must be same as num_concepts.
	
	"""

	def __init__(self, n, num_concepts, probs, query_concepts, concept_weights=None):
		self.n = n
		self.num_concepts = num_concepts
		self.probs = probs
		self.query_concepts = query_concepts
		self.concept_weights = concept_weights
		self.cpp_obj = None

		if self.n <= 0:
			raise Exception("ERROR: Number of elements in ground set must be positive")

		if self.n != len(self.probs):
			raise Exception("ERROR: Mismtach between n and len(probs)")

		if self.num_concepts != len(self.probs[0]):
			raise Exception("ERROR: Mismtach between num_concepts and len(probs[0])")
		
		if (type(self.concept_weights) != type(None)):
			if self.num_concepts != len(self.concept_weights):
			    raise Exception("ERROR: Mismtach between num_conepts and len(concept_weights)")
		else:
			self.concept_weights = [1] * self.num_concepts

		self.cpp_obj = ProbabilisticSetCoverMutualInformation(self.n, self.num_concepts, self.probs, self.concept_weights, self.query_concepts)

		self.effective_ground = set(range(n))

	