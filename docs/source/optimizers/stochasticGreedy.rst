.. _optimizers.stochastic-greedy:

Stochastic (Random) Greedy
==========================

Given a set of items :math:`V = \{1, 2, 3, \cdots, n\}` which we also call the *Ground Set*, define a utility function (set function) :math:`f:2^V \rightarrow \Re`, which measures how good a subset :math:`X \subseteq V` is. Let :math:`c :2^V \rightarrow \Re` be a cost function, which describes the cost of the set (for example, the size of the subset). The goal is then to have a subset :math:`X` which maximizes :math:`f` while simultaneously minimizing the cost function :math:`c`. It is easy to see that maximizing a generic set function becomes computationally infeasible as :math:`V` grows. Often the cost :math:`c` is budget constrained (for example, a fixed set summary) and a natural formulation of this is the following problem:

.. math::
		\max\{f(X) \mbox{ such that } c(X) \leq b\}

For any :class:`~submodlib.functions.setFunction.SetFunction`, stochastic greedy maximization can be achieved by calling :func:`~submodlib.functions.setFunction.SetFunction.maximize` with *optimizer='StochasticGreedy'*. The stochastic greedy optimizer in this library is an implementation of the stochastic greedy algorithm proposed by :cite:`mirzasoleiman2015lazier`. The main idea is to improve over naive greedy by a sub-sampling step. Specifically, in each step it first samples a set :math:`R` of size :math:`(n/k)\log(1/\epsilon)` uniformly at random and then adds that element from R to the greedy set A which increases its value the most. Such an element is added in every iteration until the desired budget (*budget*) is achieved or the best gain in any iteration is zero (*stopIfZeroGain*) or negative (*stopIfNegativeGain*).

.. note::
         Stochastic greedy optimizer has provably linear running time independent of the budget, while simultaneously having the same approximation ratio guarantee (in expectation). It is substantially faster than both naive greedy and lazy greedy optimizers.

.. note::
        At a very high level stochastic greedy's improvement over naive greedy is similar in spirit to how stochastic gradient descent improves the running time of gradient descent for convex optimization. 