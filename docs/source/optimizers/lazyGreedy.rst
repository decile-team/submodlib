.. _optimizers.lazy-greedy:

Lazy Greedy or Accelerated Greedy
=================================

Given a set of items :math:`V = \{1, 2, 3, \cdots, n\}` which we also call the *Ground Set*, define a utility function (set function) :math:`f:2^V \rightarrow \Re`, which measures how good a subset :math:`X \subseteq V` is. Let :math:`c :2^V \rightarrow \Re` be a cost function, which describes the cost of the set (for example, the size of the subset). The goal is then to have a subset :math:`X` which maximizes :math:`f` while simultaneously minimizing the cost function :math:`c`. It is easy to see that maximizing a generic set function becomes computationally infeasible as :math:`V` grows. Often the cost :math:`c` is budget constrained (for example, a fixed set summary) and a natural formulation of this is the following problem:

.. math::
		\max\{f(X) \mbox{ such that } c(X) \leq b\}

For any :class:`~submodlib.functions.setFunction.SetFunction`, lazy greedy maximization can be achieved by calling :func:`~submodlib.functions.setFunction.SetFunction.maximize` with *optimizer='LazyGreedy'*. The lazy greedy optimizer in this library is an implementation of the accelerated greedy algorithm described in :cite:`minoux1978accelerated`. Essentially, it maintains an upper bound of the marginal gain of every item and reduces them as the optimal set grows. Due to the submodularity of the function, it is guaranteed that the marginal gain of any element on a set will always be less than or equal to that on a smaller set. In any iteration, because of maintaining the upper bounds in a descending order, the algorithm doesnt have to scan the entire remaining ground set to look for the next best element to add. Thus lazy greedy optimizer is several times faster than the naive greedy optimizer. The best element is added in every iteration until the desired budget (*budget*) is achieved or the best gain in any iteration is zero (*stopIfZeroGain*) or negative (*stopIfNegativeGain*).

.. note::
        LazyGreedy optimizer will work ONLY for functions that are guaranteed to be submodular.