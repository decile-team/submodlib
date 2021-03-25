import random
import numpy as np
from submodlib import SetCoverFunction

num_concepts=3
num_samples=9
budget=5

def test():
    cover_set = []
    np.random.seed(1)
    random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        cover_set.append(set(random.sample(list(range(num_concepts)), random.randint(0,num_concepts))))
    print("Cover set: ", cover_set)
    print("Concept weights: ", concept_weights)
    obj = SetCoverFunction(n=num_samples, cover_set=cover_set, num_concepts=num_concepts, concept_weights=concept_weights)
    print("Testing SetCover's maximize")
    greedy1 = obj.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    greedy2 = obj.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    greedy3 = obj.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    greedy4 = obj.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(greedy1)
    print(greedy2)
    print(greedy3)
    print(greedy4)


if __name__ == '__main__':
    test()