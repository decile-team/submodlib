.. _functions.submodular-mutual-information:

Submodular Mutual Information
==============================

Denote :math:`\mathcal{V}` as the ground-set of items to be summarized. We denote by :math:`\mathcal{V}^{\prime}` an auxiliary set that contains user-provided information such as a query (for query-focused summarization or targeted subset selection). The auxiliary information provided by the user may not be in the same space as the items in :math:`\mathcal{V}` -- for example, if the items in :math:`\mathcal{V}` are images, the query could be text queries. In such a case, we assume we have a *joint* embedding that can represent both the query and the image items, and correspondingly, we can define similarity between the items in :math:`\mathcal{V}` and :math:`\mathcal{V}^{\prime}`. Next, let :math:`\Omega  = \mathcal{V} \cup \mathcal{V}^{\prime}` and define a set function :math:`f: 2^{\Omega} \rightarrow \Re`. Although :math:`f` is defined on :math:`\Omega`, summarization is on the items in :math:`\mathcal{V}`, i.e., the discrete optimization problem will be only on subsets of :math:`\mathcal{V}`.

We define the submodular mutual information :cite:`guillory2011-active-semisupervised-submodular,levin2020online` between two sets :math:`A,B` as 

.. math::
		I_f(A; B) = f(A) + f(B) - f(A \cup B)

It is easy to see that :math:`I_f(A; B)` is equal to the mutual information between two random variables when :math:`f` is the entropy function. Intuitively, this measures the similarity between :math:`B` and :math:`A` where :math:`B` is the query set.

.. note::
		:math:`I_f(A; B) = f(A) - f(A|B)`

Properties of submodular mutual information are studied at length in :cite:`iyer2021submodular`.

For application in query-focused summarization, :math:`B = Q` where :math:`Q \subseteq \mathcal{V}^{\prime}` is a query set.

Some simple properties of SMI which follow almost immediately from definition is that :math:`I_f(A; B) \geq 0` and :math:`I_f(A; B)` is also monotone in :math:`A` for a fixed :math:`B`. :math:`I_f(A; Q)` models the mutual coverage, or shared information, between :math:`A` and :math:`Q`, and is thus useful for modeling query relevance in query-focused summarization. 
		
.. note::
		:math:`I_f(A; Q)` is unfortunately not submodular in :math:`A` for a fixed :math:`Q` in general :cite:`krause2008near`. However some instantiations of SMI (using a some submodular function) may turn out to be submodular.

**Examples of Submodular Mutual Information functions:**

- :class:`~submodlib.functions.facilityLocationMutualInformation.FacilityLocationMutualInformationFunction`
- :class:`~submodlib.functions.facilityLocationVariantMutualInformation.FacilityLocationVariantMutualInformationFunction`
- :class:`~submodlib.functions.graphCutMutualInformation.GraphCutMutualInformationFunction`
- :class:`~submodlib.functions.logDeterminantMutualInformation.LogDeterminantMutualInformationFunction`
- :class:`~submodlib.functions.setCoverMutualInformation.SetCoverMutualInformationFunction`
- :class:`~submodlib.functions.probabilisticSetCoverMutualInformation.ProbabilisticSetCoverMutualInformationFunction`
- :class:`~submodlib.functions.concaveOverModular.ConcaveOverModularFunction`