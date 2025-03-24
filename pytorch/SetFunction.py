from typing import Set, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import random
from pytorch.optimizer.LazierThanLazyGreedyOptimizer import LazierThanLazyGreedyOptimizer
from pytorch.optimizer.LazyGreedyOptimizer import LazyGreedyOptimizer
from pytorch.optimizer.NaiveGreedyOptimizer import NaiveGreedyOptimizer
from pytorch.optimizer.StochasticGreedyOptimizer import StochasticGreedyOptimizer


class SetFunction(nn.Module):
    def __init__(self):
        pass

    def evaluate(self, X: Set[int]) -> float:
        return self.evaluate(X)

    def evaluate_with_memoization(self, X: Set[int]) -> float:
        return self.evaluate_with_memoization(X)

    def marginal_gain(self, X: Set[int], item: int) -> float:
        return self.marginal_gain(X, item)

    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
       return self.marginal_gain_with_memoization(X, item)

    def update_memoization(self, X: Set[int], item: int) -> None:
        return self.update_memoization(X, item)


    def get_effective_ground_set(self) -> Set[int]:
        return self.get_effective_ground_set()

    def maximize(self, optimizer: str, budget: float, stopIfZeroGain: bool, stopIfNegativeGain: bool, verbose: bool,
                  costs: List[float] = None, cost_sensitive_greedy: bool = False, show_progress: bool = False, epsilon: float = 0.0) -> List[Tuple[int, float]]:
        optimizer = self._get_optimizer(optimizer)
        if optimizer:
            return optimizer.maximize(self, budget, stopIfZeroGain, stopIfZeroGain, verbose, show_progress, costs, cost_sensitive_greedy)
        else:
            print("Invalid Optimizer")
            return []

    def _get_optimizer(self, optimizer_name: str):
        if optimizer_name == "NaiveGreedy":
            return NaiveGreedyOptimizer()
        # define all optimizer classed into files
        elif optimizer_name == "LazyGreedy":
            return LazyGreedyOptimizer()
        elif optimizer_name == "StochasticGreedy":
            return StochasticGreedyOptimizer()
        elif optimizer_name == "LazierThanLazyGreedy":
            return LazierThanLazyGreedyOptimizer()
        else:
            return None

    def cluster_init(self, n: int, k_dense: List[List[float]], ground: Set[int],
                     partial: bool, lambda_: float) -> None:
        self.cluster_init(n, k_dense, ground, partial, lambda_)

    def set_memoization(self, X: Set[int]) -> None:
        self.set_memoization(X)

    def clear_memoization(self) -> None:
        self.clear_memoization()
