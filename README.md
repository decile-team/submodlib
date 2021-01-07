# submodlib

submodlib is an efficient and scalable library for submodular optimization which finds its application in summarization, data subset selection, hyper parameter tuning etc. It offers great ease-of-use and flexibility in the way it can be used.

**Setup**
* `$ conda create --name submodlib python=3`
* `$ conda activate submodlib`
* `$ conda install -c conda-forge pybind11`
* `$ pip install numpy matplotlib scipy sklearn`
* `$ pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple submodlib==0.0.8`
* `$ pip install pip install ipython==7.19 notebook` (to be able to run the tutorial notebooks)

**Usage**

It is very easy to get started with submodlib. Using a submodular function in submodlib essentially boils down to just two steps:

1. instantiate the corresponding function object
2. invoke the desired method on the created object

The most frequently used methods are:
1. f.evaluate() - takes a subset and returns the score of the subset as computed by the function f
2. f.marginalGain() - takes a subset and an element and returns the marginal gain of adding the element to the subset, as computed by f
3. f.maximize() - takes a budget and an optimizer to return an optimal set as a result of maximizing f

For example,
```
from submodlib.functions.facilityLocation import FacilityLocationFunction
objFL = FacilityLocationFunction(n=43, data=data, mode="dense", metric="euclidean")
greedyList = objFL.maximize(10,'NaiveGreedy', False, False, False)
```

For a more detailed discussion on all possible usage patterns, please see [Basic Usage](https://github.com/vishkaush/submodlib/blob/master/tutorials/Basic%20Usage.ipynb)


**Functions**
* Classic sub-modular functions (models representation, diversity, importance, relevance, coverage)
    * [Facility Location](https://submodlib.readthedocs.io/en/latest/functions/facilityLocation.html)
    * [Disparity Sum](https://submodlib.readthedocs.io/en/latest/functions/disparitySum.html)
    * [DisparityMin](https://submodlib.readthedocs.io/en/latest/functions/disparityMin.html)
    * [Log Determinant](https://submodlib.readthedocs.io/en/latest/functions/logDeterminant.html)
    * [Set Cover](https://submodlib.readthedocs.io/en/latest/functions/setCover.html)
    * [Probabilistic Set Cover](https://submodlib.readthedocs.io/en/latest/functions/probabilisticSetCover.html)
    * [Graph Cut](https://submodlib.readthedocs.io/en/latest/functions/graphCut.html)
    * [Feature Based](https://submodlib.readthedocs.io/en/latest/functions/featureBased.html)
    * [Saturated Coverage](https://submodlib.readthedocs.io/en/latest/functions/saturatedCoverage.html)
* Submodular Information Measures (additionally models query-relevance, query-irrelevance, privacy-preserving)
    * [Submodular Mutual Information](https://submodlib.readthedocs.io/en/latest/functions/submodularMutualInformation.html)
    * [Conditional Gain](https://submodlib.readthedocs.io/en/latest/functions/conditionalGain.html)
    * [Conditional Mutual Information](https://submodlib.readthedocs.io/en/latest/functions/conditionalMutualInformation.html)
* [Mixture Function](https://submodlib.readthedocs.io/en/latest/functions/mixture.html)
* Supervised Subset Selection
    * [Clustered Function](https://submodlib.readthedocs.io/en/latest/functions/clustered.html)

We discuss the representational power and modeling capabilities of different functions [here](https://github.com/vishkaush/submodlib/blob/master/tutorials/Representational%20Power%20of%20Different%20Functions.ipynb)

**Optimizers**
* [Naive Greedy](https://submodlib.readthedocs.io/en/latest/optimizers/naiveGreedy.html)
* [Rnadom Greedy](https://submodlib.readthedocs.io/en/latest/optimizers/randomGreedy.html)
* [Lazy Greedy](https://submodlib.readthedocs.io/en/latest/optimizers/lazyGreedy.html)
* Lazier Than Lazy Greedy
* Distributed Greedy
* Bi Directional Greedy
* Sieve Greedy







