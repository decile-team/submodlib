.. _optimizers.lazier-than-lazy-greedy:

Lazier Than Lazy Greedy
=======================

Given a set of items :math:`V = \{1, 2, 3, \cdots, n\}` which we also call the *Ground Set*, define a utility function (set function) :math:`f:2^V \rightarrow \Re`, which measures how good a subset :math:`X \subseteq V` is. Let :math:`c :2^V \rightarrow \Re` be a cost function, which describes the cost of the set (for example, the size of the subset). The goal is then to have a subset :math:`X` which maximizes :math:`f` while simultaneously minimizing the cost function :math:`c`. It is easy to see that maximizing a generic set function becomes computationally infeasible as :math:`V` grows. Often the cost :math:`c` is budget constrained (for example, a fixed set summary) and a natural formulation of this is the following problem:

.. math::
		\max\{f(X) \mbox{ such that } c(X) \leq b\}

For any :class:`~submodlib.functions.setFunction.SetFunction`, lazier-than-lazy greedy maximization can be achieved by calling :func:`~submodlib.functions.setFunction.SetFunction.maximize` with *optimizer='LazierThanLazyGreedy'*. The implementation of lazier-than-lazy greedy optimizer in this library is an implementation of "random sampling with lazy evaluation" proposed by :cite:`mirzasoleiman2015lazier`. It combines both stochastic greedy and lazy greedy approaches. Essentially, in every iteration, it applies lazy greedy for finding the best element from a random sub sample of the remaining ground set and adds that element to the greedy set. Such an element is added in every iteration until the desired budget (*budget*) is achieved or the best gain in any iteration is zero (*stopIfZeroGain*) or negative (*stopIfNegativeGain*).

.. note::
         For submodular functions, LazierThanLazyGreedy optimizer is the most efficient, followed by StochasticGreedy, LazyGreedy and NaiveGreedy in the descending order of speed. 