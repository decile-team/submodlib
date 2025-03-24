# setCover.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>
from .setFunction import SetFunction
import torch
if torch.cuda.is_available() :
	from pytorch.submod import SetCover
else:
	from submodlib_cpp import SetCover

class SetCoverFunction(SetFunction):
	
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
		print("starting setCover.py self.cpp_obj = SetCover line 40 (at 60)")
		
		if torch.cuda.is_available() :
			self.cpp_obj = SetCover(self.n, self.cover_set, self.num_concepts, self.concept_weights)
		else:
			self.cpp_obj = SetCover(self.n, self.cover_set, self.num_concepts, self.concept_weights)
		self.effective_ground = set(range(n))
		
