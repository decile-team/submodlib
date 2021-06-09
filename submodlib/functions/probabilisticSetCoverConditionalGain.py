# probabilisticSetCoverConditionalGain.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
from submodlib_cpp import ProbabilisticSetCoverConditionalGain

class ProbabilisticSetCoverConditionalGainFunction(SetFunction):
	"""Implementation of the Probabilistic Set Cover Conditional Gain (PSCCG) function.

	Given a :ref:`functions.conditional-gain` function, Probabilistic Set Cover Conditional Gain function is its instantiation using a :class:`~submodlib.functions.probabilisticSetCover.ProbabilisticSetCoverFunction`. Mathematically, it takes the following form:

	.. math::
			f(A | P) = \\sum\\limits_{u \\in \\mathcal{U}} w_u\\bar{P_u}(A)P_u(P)
	
	Parameters
	----------

	n : int
		Number of elements in the ground set. Must be > 0.
	
	num_concepts : int
		Number of concepts.
	
	probs : list
		List of probability vectors for each data point / image, each probability vector containing the probabilities with which that data point / image covers each concept. Hence each list is num_concepts dimensional and probs contains n such lists.
	
	private_concepts : set
		Set of private concepts. That is, the concepts which should not be covered in the optimal subset.
	
	concept_weights : list
		Weight :math:`w_i` of each concept. Size must be same as num_concepts.
	
	"""

	def __init__(self, n, num_concepts, probs, private_concepts, concept_weights=None):
		self.n = n
		self.num_concepts = num_concepts
		self.probs = probs
		self.private_concepts = private_concepts
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

		self.cpp_obj = ProbabilisticSetCoverConditionalGain(self.n, self.num_concepts, self.probs, self.concept_weights, self.private_concepts)

		self.effective_ground = set(range(n))

	