# setCover.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
from submodlib_cpp import SetCover

class SetCoverFunction(SetFunction):
	"""Implementation of the Set-Cover (SC) submodular function.
	
	For a subset :math:`A`, its Set Cover evaluation is defined as: 

	.. math::
			f(A) = w(\\cup_{a \\in A} \\gamma(a)) = w(\\gamma(A))

	where :math:`\\gamma(A)` refers to the set of concepts covered by :math:`A`. Thus the set of all concepts :math:`\\mathcal{U} = \\gamma(\\mathcal{V})`. :math:`w` is a weight vector in :math:`\\Re^{|\\mathcal{U}|}`. Intuitively, each element in :math:`\\mathcal{V}` *covers* a set of elements from the concept set :math:`U` and hence :math:`w(\\gamma(A))` is total weight of concepts covered by elements in :math:`A`. Note that :math:`\\gamma(A \\cup B) = \\gamma(A) \\cup \\gamma(B)` and hence :math:`f(A \\cup B) = w(\\gamma(A \\cup B)) = w(\\gamma(A) \\cup \\gamma(B))`.

	Alternatively we can also view the function as follows. With :math:`U` being the set of all concepts (namely :math:`U = \\gamma(\\mathcal{V})`) and :math:`c_u(i)` denoting whether the concept :math:`u \\in U` is covered by the element :math:`i \\in \\mathcal{V}` i.e :math:`c_u(i) = 1` if :math:`u \\in \\gamma(\\{i\\})` and is zero otherwise. We then define :math:`c_u(A) = \\sum_{a\\in A} c_u(a)` as the count of concept :math:`u` in set :math:`A`, and the weighted set cover can then be written as:
		
	.. math::
			f(A) = \\sum_{u \\in U} w_u \\min(c_u(A), 1)
			
	.. note::
			Set Cover functions models coverage of concepts and is monotone submodular.

	Parameters
	----------
	n : int
		Number of elements in the ground set, must be > 0.

	cover_set : list
		List of sets. Each set is the set of concepts covered by the corresponding data point / image. Hence cover_set is of size n.

	num_concepts : int
		Number of concepts.

	concept_weights : list
		Weight :math:`w_i` of each concept. Size must be same as num_concepts.

	"""

	def __init__(self, n, cover_set, num_concepts, concept_weights=None):
		self.n = n
		self.cover_set = cover_set
		self.num_concepts = num_concepts
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

		self.cpp_obj = SetCover(self.n, self.cover_set, self.num_concepts, self.concept_weights)

		self.effective_ground = set(range(n))

	