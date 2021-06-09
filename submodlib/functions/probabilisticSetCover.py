# probabilisticSetCover.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
from submodlib_cpp import ProbabilisticSetCover

class ProbabilisticSetCoverFunction(SetFunction):
	"""Implementation of the Probabilistic Set Cover (PSC) submodular function.
	
	This variant of the :class:`~submodlib.functions.setCover.SetCoverFunction` is defined as 
	
	.. math::
			f_{psc}(X) = \\sum_{u \\in \\mathcal{U}} w_u(1 - P_u(X))
	
	where :math:`\\mathcal{U}` is the set of concepts, :math:`w_u` is the weight of the concept :math:`u` and :math:`P_u(X) = \\prod_{j \\in X} (1 - p_{uj})` where :math:`p_{xu}` is the probability with which concept :math:`u` is covered by element :math:`x`. Thus, :math:`P_u(X)` is the probability that :math:`X` *doesn't* cover concept :math:`u`. In other words,

	.. math::
			f_{psc}(X) = \\sum_{u \\in \\mathcal{U}} w_u(1 - \\prod_{x \\in X} (1 - p_{xu}))

	Intuitively, PSC is a soft version of the SC, which allows for probability of covering concepts, instead of a binary yes/no, as is the case with SC.

	Similar to the set cover function, this function models the coverage aspect of the candidate summary (subset), viewed stochastically and is also monotone submodular.

	Parameters
	----------
	n : int
		Number of elements in the ground set, must be > 0.
	
	probs : list
		List of probability vectors for each data point / image, each probability vector containing the probabilities with which that data point / image covers each concept. Hence each list is num_concepts dimensional and probs contains n such lists.

	num_concepts : int
		Number of concepts.

	concept_weights : list
		Weight :math:`w_i` of each concept. Size must be same as num_concepts.
	
	"""

	def __init__(self, n, probs, num_concepts, concept_weights=None):
		self.n = n
		self.probs = probs
		self.num_concepts = num_concepts
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

		self.cpp_obj = ProbabilisticSetCover(self.n, self.probs, self.num_concepts, self.concept_weights)

		self.effective_ground = set(range(n))