import random
import math

class LazierThanLazyGreedyOptimizer:
    def __init__(self):
        pass

    @staticmethod
    def equals(val1, val2, eps):
        return abs(val1 - val2) < eps

    @staticmethod
    def print_sorted_set(sorted_set):
        print("[", end="")
        for val, elem in sorted_set:
            print(f"({val}, {elem}), ", end="")
        print("]")

    def maximize(self, f_obj, budget, stop_if_zero_gain=False, stop_if_negative_gain=False,
                    epsilon=0.1, verbose=False, show_progress=False, costs=None, cost_sensitive_greedy=False):
            greedy_vector = []
            greedy_set = set()

            if costs is None:
                greedy_vector.reserve(budget)
                greedy_set.reserve(budget)

            rem_budget = budget
            remaining_set = set(f_obj.get_effective_ground_set())
            n = len(remaining_set)
            epsilon = 0.05
            random_set_size = int((n / budget) * math.log(1 / epsilon))

            if verbose:
                print(f"Epsilon = {epsilon}")
                print(f"Random set size = {random_set_size}")
                print("Ground set:")
                print(remaining_set)
                print(f"Num elements in ground set = {len(remaining_set)}")
                print("Starting the LazierThanLazy greedy algorithm")
                print("Initial greedy set:")
                print(greedy_set)

            f_obj.clear_memoization()
            best_id = None
            best_val = None

            i = 0
            step = 1
            display_next = step
            percent = 0
            N = rem_budget
            iter_count = 0

            while rem_budget > 0:
                random_set = set()
                while len(random_set) < random_set_size:
                    elem = random.randint(0, n - 1)
                    if elem in remaining_set and elem not in random_set:
                        random_set.add(elem)

                if verbose:
                    print(f"Iteration {i}")
                    print(f"Random set = {random_set}")
                    print("Now running lazy greedy on the random set")

                candidate_id = None
                candidate_val = None
                new_candidate_bound = None

                # Compute gains only for the elements in the remaining set
                gains = [(f_obj.marginal_gain_with_memoization(greedy_set, elem, False), elem)
                        for elem in remaining_set]

                for j, (val, elem) in enumerate(sorted(gains, key=lambda x: (-x[0], x[1]))):
                    if elem in random_set and elem not in greedy_set:  # Check if the element is not already selected
                        if verbose:
                            print(f"Checking {elem}...")
                        candidate_id = elem
                        candidate_val = val
                        new_candidate_bound = f_obj.marginal_gain_with_memoization(greedy_set, candidate_id, False)
                        if verbose:
                            print(f"Updated gain as per updated greedy set = {new_candidate_bound}")
                        next_elem = gains[j + 1] if j + 1 < len(gains) else None
                        if new_candidate_bound >= next_elem[0] if next_elem else float('-inf'):
                            if verbose:
                                print("..better than next best upper bound, "
                                      "selecting...")
                            best_id = candidate_id
                            best_val = new_candidate_bound
                            break

                if verbose:
                    print(f"Next best item to add is {best_id} and its value addition is {best_val}")

                remaining_set.remove(best_id)

                if (best_val < 0 and stop_if_negative_gain) or (self.equals(best_val, 0, 1e-5) and stop_if_zero_gain):
                    break
                else:
                    f_obj.update_memoization(greedy_set, best_id)
                    greedy_set.add(best_id)
                    greedy_vector.append((best_id, best_val))
                    rem_budget -= 1

                    if verbose:
                        print(f"Added element {best_id} and the gain is {best_val}")
                        print("Updated greedy set:", greedy_set)

                    if show_progress:
                        percent = int(((iter_count + 1.0) / N) * 100)
                        if percent >= display_next:
                            print("\r", "[" + "|" * (percent // 5) + " " * (100 // 5 - percent // 5) + "]", end="")
                            print(f" {percent}% [Iteration {iter_count + 1} of {N}]", end="")
                            display_next += step
                        iter_count += 1

                i += 1

            return greedy_vector
