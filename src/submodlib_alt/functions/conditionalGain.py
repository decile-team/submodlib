# conditionalGain.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>

from .setFunction import SetFunction

class ConditionalGainFunction(SetFunction):
	"""Implementation of the Conditional Gain (CG) of a submodular function.
	
	Denote :math:`V` as the ground-set of items to be summarized. We denote by :math:`V^{\\prime}` an auxiliary set that contains user-provided information such as a query (for query-focused summarization) or a private set (for privacy-preserving summarization). The auxiliary information provided by the user may not be in the same space as the items in :math:`V` -- for example, if the items in :math:`V` are images, the query could be text queries. In such a case, we assume we have a *joint* embedding that can represent both the query and the image items, and correspondingly, we can define similarity between the items in :math:`V` and :math:`V^{\\prime}`. Next, let :math:`\\Omega  = V \\cup V^{\\prime}` and define a set function :math:`f: 2^{\\Omega} \\rightarrow \\mathbf{R}`. Although :math:`f` is defined on :math:`\\Omega`, summarization is on the items in :math:`V`, i.e., the discrete optimization problem will be only on subsets of :math:`V`.

	Given a set of items :math:`A, B \\subseteq \\Omega`, the conditional gain is the gain in function value by adding :math:`B` to :math:`A`

	.. math::
			f(A | B) = f(A \\cup B) - f(B)
	
	When :math:`f` is entropy, this corresponds to the conditional entropy. Examples of CG include :math:`f(A | P) = f(A \\cup P) - f(P), A \\subseteq V` where :math:`P \\subseteq V^{\\prime}` is either the *private set* or the *irrelevant set*.

	Another example of CG is :math:`f(A | A_0), A, A_0 \\in V` where :math:`A_0` is a summary chosen by the user *before*. This is important for update summarization where the desired summary should be different from a preexisting one.

	The conditional gain has been studied in a number of optimization problems involving submodular functions :cite:`iyer2012algorithms,iyer2013submodularScsk,krause2014submodular`.

	Properties of conditional gain are studied at length in :cite:`iyer2020submodular`.

	Parameters
	----------
	
	f : SetFunction
		The particular instantiated set function to be used for instantiating this Conditional Gain function
	
	b : set
		The :math:`B` set as defined above. For example, it could be the private set in case of privacy preserving summarization or existing subset in case of update summarization	
	
	"""

	def __init__(self, f, b):
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