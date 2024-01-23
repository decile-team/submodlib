import random
from typing import List, Tuple, Set
import math
import sys

class StochasticGreedyOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def equals(val1: float, val2: float, eps: float) -> bool:
        return abs(val1 - val2) < eps

    def maximize(self, f_obj, budget: float, stop_if_zero_gain: bool,
                 stop_if_negative_gain: bool, epsilon: float = 1, verbose: bool = True,
                 show_progress: bool = False, costs: List[float] = None, cost_sensitive_greedy: bool = False) -> List[Tuple[int, float]]:
        # TODO: Implement handling of equal guys and different sizes of each item later
        # TODO: Implement cost-sensitive selection

        greedy_vector = []
        greedy_set = set()

        # if not costs:
        #     # Every element is of the same size, budget corresponds to cardinality
        #     greedy_vector.reserve(budget)
        #     greedy_set.reserve(budget)

        rem_budget = budget
        remaining_set = set(f_obj.get_effective_ground_set())
        n = len(remaining_set)
        epsilon = 0.05
        random_set_size = int((n / budget) * math.log(1 / epsilon))
        if verbose:
            print(f"Epsilon = {epsilon}")
            print(f"Random set size = {random_set_size}")
            print("Ground set:")
            print(" ".join(map(str, remaining_set)))
            print(f"Num elements in groundset = {len(remaining_set)}")
            print("Starting the stochastic greedy algorithm")
            print("Initial greedy set:")
            print(" ".join(map(str, greedy_set)))

        f_obj.clear_memoization()
        random.seed(1)
        best_id = -1
        best_val = -1 * float('inf')
        i = 0
        step = 1
        display_next = step
        percent = 0
        N = rem_budget
        iter = 0

        while rem_budget > 0:
            random_set = set()
            while len(random_set) < random_set_size:
                elem = random.randint(0, n - 1)
                if elem in remaining_set and elem not in random_set:
                    random_set.add(elem)

            if verbose:
                print(f"Iteration {i}")
                print(f"Random set = {list(random_set)}")
                print("Now running naive greedy on the random set")

            best_id = -1
            best_val = -1 * float('inf')

            for elem in random_set:
                gain = f_obj.marginal_gain_with_memoization(greedy_set, elem, False)
                if gain > best_val:
                    best_id = elem
                    best_val = gain

            if verbose:
                if best_id == -1:
                    raise ValueError("Nobody had greater gain than minus infinity!!")
                print(f"Next best item to add is {best_id} and its value addition is {best_val}")

            if (best_val < 0 and stop_if_negative_gain) or (self.equals(best_val, 0, 1e-5) and stop_if_zero_gain):
                break
            else:
                f_obj.update_memoization(greedy_set, best_id)
                greedy_set.add(best_id)
                greedy_vector.append((best_id, best_val))
                rem_budget -= 1
                remaining_set.remove(best_id)

                if verbose:
                    print(f"Added element {best_id} and the gain is {best_val}")
                    print("Updated greedy set:", " ".join(map(str, greedy_set)))

                if show_progress:
                    percent = int(((iter + 1.0) / N) * 100)
                    if percent >= display_next:
                        print(f"\r[{'|' * (percent // 5)}{' ' * (100 // 5 - percent // 5)}]", end="")
                        print(f"{percent}% [Iteration {iter + 1} of {N}]", end="")
                        sys.stdout.flush()
                        display_next += step
                    iter += 1

            i += 1

        return greedy_vector
