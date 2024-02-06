import torch
import torch.nn as nn
import numpy as np
import random
from ..SetFunction import SetFunction

class SetCoverFunction(SetFunction):
    def __init__(self, n, cover_set, num_concepts, concept_weights = None):
        super(SetFunction, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n = n
        self.cover_set = cover_set
        self.num_concepts = num_concepts
        self.concept_weights = concept_weights
        if self.concept_weights is None:
            self.concept_weights = [1.0] * num_concepts
        else:
            self.concept_weights = torch.tensor(concept_weights, dtype=torch.float32).to(device)


        self.concepts_covered_by_x = set()


    def evaluate(self, X):
      result = 0.0

      if X.numel() == 0:
          return 0.0

      concepts_covered = set()
      for elem in X:
          concepts_covered.update(self.cover_set[elem.item()])

      for con in concepts_covered:
          result += self.concept_weights[con]

      return result


    def evaluate_with_memoization(self, X):
        result = 0.0

        if X.numel() == 0:
            print("hi")
            return 0.0

        for con in self.concepts_covered_by_x:
            result += self.concept_weights[con]
            print(result)

        return result

    def marginal_gain(self, X, item):
        gain = 0.0

        if item in X:
            return 0.0

        concepts_covered = set()
        for elem in X:
            concepts_covered.update(self.cover_set[elem])

        for con in self.cover_set[item]:
            if con not in concepts_covered:
                gain += self.concept_weights[con]

        return gain.item()

    def marginal_gain_with_memoization(self, X, item, enable_checks=True):
        gain = 0.0

        if enable_checks and item in X:
            return 0.0
        for con in self.cover_set[item]:
          if con not in self.concepts_covered_by_x:
                gain += self.concept_weights[con]

        return gain

    def update_memoization(self, X, item):
        if item in X:
            return

        self.concepts_covered_by_x.update(self.cover_set[item])

    def get_effective_ground_set(self):
        return set(range(self.n))

    def clear_memoization(self):
        self.concepts_covered_by_x.clear()

    def set_memoization(self, X):
        self.clear_memoization()
        temp = set()
        for elem in X:
            self.update_memoization(temp, elem)
            temp.add(elem)
