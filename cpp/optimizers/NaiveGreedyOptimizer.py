import torch
import random
from typing import List, Tuple, Set

class NaiveGreedyOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def equals(val1, val2, eps):
        return abs(val1 - val2) < eps

    def maximize(
        self, f_obj, budget, stop_if_zero_gain, stopIfNegativeGain, verbose, show_progress, costs, cost_sensitive_greedy
    ):
        greedy_vector = []
        greedy_set = set()
        if not costs:
            # greedy_vector = [None] * budget
            greedy_set = set()
        rem_budget = budget
        ground_set = f_obj.get_effective_ground_set()
        #print(ground_set)
        if verbose:
            print("Ground set:")
            print(ground_set)
            print(f"Num elements in groundset = {len(ground_set)}")
            print("Costs:")
            print(costs)
            print(f"Cost sensitive greedy: {cost_sensitive_greedy}")
            print("Starting the naive greedy algorithm")
            print("Initial greedy set:")
            print(greedy_set)

        f_obj.clear_memoization()
        best_id = None
        best_val = None
        step = 1
        display_next = step
        percent = 0
        N = rem_budget
        iter_count = 0

        while rem_budget > 0:
            best_id = None
            best_val = float("-inf")

            for i in ground_set:
                if i in greedy_set:
                    continue
                gain = f_obj.marginal_gain_with_memoization(greedy_set, i, False)
                # print(gain)
                if verbose:
                    print(f"Gain of {i} is {gain}")

                if gain > best_val:
                    best_id = i
                    best_val = gain

            if verbose:
                print(f"Next best item to add is {best_id} and its value addition is {best_val}")

            if (best_val < 0 and stopIfNegativeGain) or (
                self.equals(best_val, 0, 1e-5) and stop_if_zero_gain
            ):
                break
            else:
                f_obj.update_memoization(greedy_set, best_id)
                greedy_set.add(best_id)
                greedy_vector.append((best_id, best_val))
                rem_budget -= 1

                if verbose:
                    print(f"Added element {best_id} and the gain is {best_val}")
                    print(f"Updated greedy set: {greedy_set}")

                if show_progress:
                    percent = int((iter_count + 1.0) / N * 100)

                    if percent >= display_next:
                        print(
                            f"\r[{'|' * (percent // 5)}{' ' * (100 // 5 - percent // 5)}]",
                            end="",
                        )
                        print(f"{percent}% [Iteration {iter_count + 1} of {N}]", end="")
                        display_next += step

                    iter_count += 1

        return greedy_vector
