import torch
from typing import List, Set, Tuple
from ..SetFunction import SetFunction

class ProbabilisticSetCover(SetFunction):
    def __init__(self, n: int, ground_set_concept_probabilities: List[List[float]], num_concepts: int, concept_weights: List[float] = None):
        super(SetFunction, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n = n
        self.ground_set_concept_probabilities = ground_set_concept_probabilities
        self.num_concepts = num_concepts
        self.concept_weights = concept_weights

        if self.concept_weights is None:
            self.concept_weights = [1.0] * num_concepts
        else:
            self.concept_weights = torch.tensor(concept_weights, dtype=torch.float32).to(device)
        self.prob_of_concepts_covered_by_X = num_concepts

    def evaluate(self, X: Set[int]) -> float:
        result = 0
        if not X:
            return result

        for i in range(self.num_concepts):
            product = 1
            for elem in X:
                product *= (1 - self.ground_set_concept_probabilities[elem][i])
            result += self.concept_weights[i] * (1 - product)

        return result

    def evaluate_with_memoization(self, X: Set[int]) -> float:
        result = 0
        if not X:
            return result

        for i in range(self.num_concepts):
            result += self.concept_weights[i] * (1 - self.prob_of_concepts_covered_by_X[i])

        return result

    def marginal_gain(self, X: Set[int], item: int) -> float:
        gain = 0
        if item in X:
            return gain

        for i in range(self.num_concepts):
            old_concept_prod = 1
            for elem in X:
                old_concept_prod *= (1 - self.ground_set_concept_probabilities[elem][i])
            gain += self.concept_weights[i] * old_concept_prod * self.ground_set_concept_probabilities[item][i]
        return gain

    def marginal_gain_with_memoization(self, X: Set[int], item: int, enable_checks: bool = True) -> float:
        gain = 0
        if enable_checks and item in X:
            return gain
        for i in range(self.num_concepts):
            gain += self.concept_weights[i] * self.prob_of_concepts_covered_by_X[i] * self.ground_set_concept_probabilities[item][i]
        return gain

    def update_memoization(self, X: Set[int], item: int):
        if item in X:
            return

        for i in range(self.num_concepts):
            self.prob_of_concepts_covered_by_X[i] *= (1 - self.ground_set_concept_probabilities[item][i])

    def get_effective_ground_set(self) -> Set[int]:
        return set(range(self.n))

    def clear_memoization(self):
        self.prob_of_concepts_covered_by_X = torch.ones(self.num_concepts, dtype=torch.double)

    def set_memoization(self, X: Set[int]):
        self.clear_memoization()
        temp = set()
        for elem in X:
            self.update_memoization(temp, elem)
            temp.add(elem)
