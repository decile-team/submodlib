.. _functions.submodular-functions:

Submodular Functions
====================

Submodularity  is a rich yet tractable sub-field of non-linear combinatorial optimization which ensures tractable algorithms :cite:`krause2014submodular` and nice connections to convexity and concavity :cite:`bach2011learning,lovasz1983submodular,iyer2015polyhedral`.

Let :math:`\mathcal{V}` denote the *ground-set* of :math:`n` data points :math:`\mathcal{V} = {1, 2, 3,...,n}` and a set function :math:`f: 2^{\mathcal{V}} \xrightarrow{} \Re`. The function :math:`f` is submodular :cite:`fujishige2005submodular` if it satisfies the diminishing marginal returns, namely

.. math::
        f(j | \mathcal{X}) \geq f(j | \mathcal{Y})

for all :math:`\mathcal{X} \subseteq \mathcal{Y} \subseteq \mathcal{V}, j \notin \mathcal{Y}`. 

Submodular functions exhibit a property that intuitively formalizes the idea of *diminishing returns*. That is, adding some instance :math:`x` to the set :math:`\mathcal{X}` provides more gain in evaluation than adding :math:`x` to a larger set :math:`\mathcal{Y}`.  Informally, since :math:`\mathcal{Y}` is a superset of :math:`\mathcal{X}` and already contains more information, adding :math:`x` will not help as much. Using a greedy algorithm to optimize a submodular function (for selecting a subset) gives a lower-bound performance guarantee of around 63\% of optimal :cite:`nemhauser1978analysis` to the above problem, and in practice these greedy solutions are often within 98\% of optimal :cite:`krause2008optimizing`.

.. note::
		Conditional Gain functions are non-negative and monotone in one argument with the other fixed :cite:`levin2020online,iyer2021submodular`.
	
Examples of Conditional Gain functions:

- :class:`~submodlib.functions.facilityLocationConditionalGain.FacilityLocationConditionalGainFunction`
- :class:`~submodlib.functions.graphCutConditionalGain.GraphCutConditionalGainFunction`
- :class:`~submodlib.functions.logDeterminantConditionalGain.LogDeterminantConditionalGainFunction`
- :class:`~submodlib.functions.setCoverConditionalGain.SetCoverConditionalGainFunction`
- :class:`~submodlib.functions.probabilisticSetCoverConditionalGain.ProbabilisticSetCoverConditionalGainFunction`