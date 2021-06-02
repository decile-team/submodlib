.. _functions.conditional-gain:

Conditional Gain
================

Denote :math:`\mathcal{V}` as the ground-set of items to be summarized. We denote by :math:`\mathcal{V}^{\prime}` an auxiliary set that contains user-provided information like a private set (for privacy-preserving summarization or query-irrelevant summarization). The auxiliary information provided by the user may not be in the same space as the items in :math:`\mathcal{V}` -- for example, if the items in :math:`\mathcal{V}` are images, the private instances could be textual topics. In such a case, we assume we have a *joint* embedding that can represent both the private instances and the image items, and correspondingly, we can define similarity between the items in :math:`\mathcal{V}` and :math:`\mathcal{V}^{\prime}`. Next, let :math:`\Omega  = \mathcal{V} \cup \mathcal{V}^{\prime}` and define a set function :math:`f: 2^{\Omega} \rightarrow \Re`. Although :math:`f` is defined on :math:`\Omega`, summarization is on the items in :math:`\mathcal{V}`, i.e., the discrete optimization problem will be only on subsets of :math:`\mathcal{V}`.

Given a set of items :math:`A, B \subseteq \Omega`, the conditional gain is the gain in function value by adding :math:`A` to :math:`B`. That is,

.. math::
		f(A | B) = f(A \cup B) - f(B)
	
When :math:`f` is entropy, this corresponds to the conditional entropy. Intuitively, :math:`f(A|B)` measures how different :math:`A` is from :math:`B`, where :math:`B` is the conditioning set or the private set or the *irrelevant* set.

Examples of CG include :math:`f(A | P) = f(A \cup P) - f(P), A \subseteq \mathcal{V}` where :math:`P \subseteq \mathcal{V}^{\prime}` is either the *private set* or the *irrelevant set*.

Another example of CG is :math:`f(A | A_0), A, A_0 \in \mathcal{V}` where :math:`A_0` is a summary chosen by the user *before*. This is important for update-summarization :cite:`dang2008overview,delort2012dualsum,li2015improving` where the desired summary should be different from a pre-existing one.

The conditional gain has been studied in a number of optimization problems involving submodular functions :cite:`iyer2012algorithms,iyer2013submodularScsk,krause2014submodular`.

Properties of conditional gain are studied at length in :cite:`iyer2021submodular`.

.. note::
		Conditional Gain functions are non-negative and monotone in one argument with the other fixed :cite:`levin2020online,iyer2021submodular`.
	
**Examples of Conditional Gain functions:**

- :class:`~submodlib.functions.facilityLocationConditionalGain.FacilityLocationConditionalGainFunction`
- :class:`~submodlib.functions.graphCutConditionalGain.GraphCutConditionalGainFunction`
- :class:`~submodlib.functions.logDeterminantConditionalGain.LogDeterminantConditionalGainFunction`
- :class:`~submodlib.functions.setCoverConditionalGain.SetCoverConditionalGainFunction`
- :class:`~submodlib.functions.probabilisticSetCoverConditionalGain.ProbabilisticSetCoverConditionalGainFunction`