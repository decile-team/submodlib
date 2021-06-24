.. _optimizers.naive-greedy:

Naive Greedy
============

Given a set of items :math:`V = \{1, 2, 3, \cdots, n\}` which we also call the *Ground Set*, define a utility function (set function) :math:`f:2^V \rightarrow \Re`, which measures how good a subset :math:`X \subseteq V` is. Let :math:`c :2^V \rightarrow \Re` be a cost function, which describes the cost of the set (for example, the size of the subset). The goal is then to have a subset :math:`X` which maximizes :math:`f` while simultaneously minimizing the cost function :math:`c`. It is easy to see that maximizing a generic set function becomes computationally infeasible as :math:`V` grows. Often the cost :math:`c` is budget constrained (for example, a fixed set summary) and a natural formulation of this is the following problem:

.. math::
		\max\{f(X) \mbox{ such that } c(X) \leq b\}

For any :class:`~submodlib.functions.setFunction.SetFunction`, naive greedy maximization can be achieved by calling :func:`~submodlib.functions.setFunction.SetFunction.maximize` with *optimizer='NaiveGreedy'*. The naive greedy optimizer implementation in this library implements the standard greedy algorithm :cite:`minoux1978accelerated`. It starts with an empty set and in every iteration adds to it a new element from the ground set with maximum marginal gain until the desired budget (*budget*) is achieved or the best gain in any iteration is zero (*stopIfZeroGain*) or negative (*stopIfNegativeGain*). The solution thus obtained is called a *greedy solution*. 

.. note::
		Unless the marginal gain of the element at each step is unique, the greedy solution will not be unique. In such a case, the current implementation adds the first best element encountered at every iteration. As unordered sets are used to represent the ground sets, this ordering need not be unique.

.. note::
		When :math:`f` is a submodular function, using a simple greedy algorithm to compute the above gives a lower-bound performance guarantee of around 63% of optimal :cite:`nemhauser1978analysis` and in practice these greedy solutions are often within 90% of optimal :cite:`krause2008optimizing`