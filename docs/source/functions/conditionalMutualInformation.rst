.. _functions.conditional-mutual-information:

Conditional Mutual Information
==============================

Denote :math:`\mathcal{V}` as the ground-set of items to be summarized. We denote by :math:`\mathcal{V}^{\prime}` an auxiliary set that contains user-provided information such as a query (for query-focused summarization) or a private set (for privacy-preserving summarization) or both in the case of joint query and privacy-preserving summarization. The auxilliary information provided by the user may not be in the same space as the items in :math:`\mathcal{V}` -- for example, if the items in :math:`\mathcal{V}` are images, the query could be text queries. In such a case, we assume we have a *joint* embedding that can represent both the query and the image items, and correspondingly, we can define similarity between the items in :math:`\mathcal{V}` and :math:`\mathcal{V}^{\prime}`. Next, let :math:`\Omega  = \mathcal{V} \cup \mathcal{V}^{\prime}` and define a set function :math:`f: 2^{\Omega} \rightarrow \Re`. Although :math:`f` is defined on :math:`\Omega`, summarization is on the items in :math:`\mathcal{V}`, i.e., the discrete optimization problem will be only on subsets of :math:`\mathcal{V}`.

We define the submodular conditional mutual information of sets :math:`A,B` given set :math:`C` as 

.. math::
		I_f(A; B | C) &= f(A | C) + f(B | C) - f(A \cup B | C) \\&= f(A \cup C) + f(B \cup C) - f(A \cup B \cup C) - f(C)

Intuitively CMI jointly models the mutual similarity between :math:`A` and :math:`B` and their collective dissimilarity from :math:`C`. 

One possible application of CMI is joint query-focused and privacy preserving summarization. Given a query set :math:`Q` and a private set :math:`P`, we would like to select a subset :math:`A \subseteq \mathcal{V}` which has a high similarity with respect to a query set :math:`Q`, while simultaneously being different from the private set :math:`P`. A natural way to do this is by maximizing the above conditional submodular mutual information such that :math:`B=Q` and :math:`C=P`.

Properties of conditional mutual information are studied at length in :cite:`iyer2021submodular`. In particular, CMI is non-negative and monotone in one argument with the other fixed :cite:`levin2020online,iyer2021submodular`. CMI however is not necessarily submodular in one argument (with the others fixed) :cite:`krause2008near, iyer2021submodular`. However, certain instantiations of CMI may turn out to be submodular.

.. note::
		Given a submodular function :math:`f`, the CMI can either be viewed as the mutual information of the conditional gain function, or the conditional gain of the submodular mutual information :cite:`kaushal2020unified`.

**Examples of Conditional Mutual Information functions:**

- :class:`~submodlib.functions.facilityLocationConditionalMutualInformation.FacilityLocationConditionalMutualInformationFunction`
- :class:`~submodlib.functions.logDeterminantConditionalMutualInformation.LogDeterminantConditionalMutualInformationFunction`
- :class:`~submodlib.functions.setCoverConditionalMutualInformation.SetCoverConditionalMutualInformationFunction`
- :class:`~submodlib.functions.probabilisticSetCoverConditionalMutualInformation.ProbabilisticSetCoverConditionaMutualInformationFunction`
