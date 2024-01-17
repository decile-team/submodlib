import torch
import heapq

class LazyGreedyOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def equals(val1, val2, eps):
        return abs(val1 - val2) < eps

    def maximize(self, f_obj, budget, stop_if_zero_gain, stop_if_negative_gain,
                 verbose, show_progress, costs, cost_sensitive_greedy):
        greedy_vector = []
        greedy_set = set()

        # if not costs:
        #     greedy_vector.reserve(budget)
        #     greedy_set.reserve(budget)

        rem_budget = budget
        ground_set = f_obj.get_effective_ground_set()

        if verbose:
            print("Ground set:")
            print(ground_set)
            print(f"Num elements in groundset = {len(ground_set)}")
            print("Costs:")
            print(costs)
            print(f"Cost sensitive greedy: {cost_sensitive_greedy}")
            print("Starting the lazy greedy algorithm")
            print("Initial greedy set:")
            print(greedy_set)

        f_obj.clear_memoization()

        container = []
        heapq.heapify(container)
        max_heap = container

        if cost_sensitive_greedy:
            for elem in ground_set:
                gain = f_obj.marginal_gain_with_memoization(greedy_set, elem, False) / costs[elem]
                heapq.heappush(max_heap, (-gain, elem))
        else:
            for elem in ground_set:
                gain = f_obj.marginal_gain_with_memoization(greedy_set, elem, False)
                heapq.heappush(max_heap, (-gain, elem))

        if verbose:
            print("Max heap constructed")

        step = 1
        display_next = step
        percent = 0
        N = rem_budget
        iter = 0

        while rem_budget > 0 and max_heap:
            current_max = heapq.heappop(max_heap)
            current_max_gain, current_max_elem = -current_max[0], current_max[1]

            if verbose:
                print(f"currentMax element: {current_max_elem} and its upper bound: {current_max_gain}")

            new_max_bound = f_obj.marginal_gain_with_memoization(greedy_set, current_max_elem, False)

            if verbose:
                print(f"newMaxBound: {new_max_bound}")

            if new_max_bound >= -max_heap[0][0]:
                if (new_max_bound < 0 and stop_if_negative_gain) or \
                        (self.equals(new_max_bound, 0, 1e-5) and stop_if_zero_gain):
                    break
                else:
                    f_obj.update_memoization(greedy_set, current_max_elem)
                    greedy_set.add(current_max_elem)
                    greedy_vector.append((current_max_elem, new_max_bound))
                    rem_budget -= 1

                    if verbose:
                        print(f"Added element {current_max_elem} and the gain is {new_max_bound}")
                        print("Updated greedySet:", greedy_set)

                    if show_progress:
                        percent = int(((iter + 1.0) / N) * 100)

                        if percent >= display_next:
                            print(f"\r[{'|' * (percent // 5)}{' ' * (100 // 5 - percent // 5)}]",
                                  end=f" {percent}% [Iteration {iter + 1} of {N}]")
                            display_next += step

                        iter += 1
            else:
                heapq.heappush(max_heap, (-new_max_bound, current_max_elem))

        return greedy_vector
