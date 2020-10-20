# optimizers.py
# Author: Vishal Kaushal <vishal.kaushal@gmail.com>

class Optimizer():
    """Base class for all optimizers

    """

    def __init__():
        pass
    
    def run():
        pass

class NaiveGreedy(Optimizer):
    """Implements the naive greedy algorithm for optimization

    Given a set :math:`V = \\{1, 2, 3, \\cdots, n\\}` of items which we also call the *Ground Set*, define a utility function (set function) :math:`f:2^V \\rightarrow \\mathbf{R}`, which measures how good a subset "math:`X \\subseteq V` is. Let :math:`c :2^V \\rightarrow \\mathbf{R}` be a cost function, which describes the cost of the set (for example, the size of the subset). The goal is then to have a subset :math:`X` which maximizes :math:`f` while simultaneously minimizing the cost function :math:`c`. It is easy to see that maximizing a generic set function becomes computationally infeasible as :math:`V` grows. Often the cost :math:`c` is budget constrained (for example, a fixed set summary) and a natural formulation of this is the following problem:

    .. math::
            \\max\\{f(X) \\mbox{ such that } c(X) \\leq b\\}
    
    When :math:`f` is a submodular function, using a simple greedy algorithm to compute the above gives a lower-bound performance guarantee of around 63% of optimal :cite:`nemhauser1978analysis` and in practice these greedy solutions are often within 90% of optimal :cite:`krause2008optimizing`

    """

    def __init__():
        pass

    def run():
        pass

class RandomGreedy(Optimizer):
    """Implements the random greedy algorithm for optimization

    

    """

    def __init__():
        pass

    def run():
        pass

class LazyGreedy(Optimizer):
    """Implements the lazy greedy algorithm for optimization

    

    """

    def __init__():
        pass

    def run():
        pass