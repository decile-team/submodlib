import pytest
import math
from sklearn.datasets import make_blobs
import numpy as np
import random
from submodlib import FacilityLocationFunction
from submodlib import FeatureBasedFunction
from submodlib import DisparitySumFunction
from submodlib import DisparityMinFunction
from submodlib import GraphCutFunction
from submodlib import ClusteredFunction
from submodlib import LogDeterminantFunction
from submodlib import SetCoverFunction
from submodlib import ProbabilisticSetCoverFunction
from submodlib.helper import create_kernel
from submodlib_cpp import FeatureBased

#allKernelFunctions = ["FacilityLocation", "DisparitySum", "GraphCut", "DisparityMin", "LogDeterminant"]
allKernelFunctions = ["LogDeterminant"]
clusteredModeFunctions = ["FacilityLocation"]
#optimizerTests = ["FacilityLocation", "GraphCut", "LogDeterminant"]
optimizerTests = ["LogDeterminant"]

#########Available markers############
# clustered_mode - for clustered mode related test cases
# fb_opt - for optimizer tests of FB functions
# fb_regular - for regular tests of FB functions
# sc_opt - for optimizer tests of SC functions
# sc_regular - for regular tests of PSC functions
# psc_opt -for optimizer tests of PSC functions
# psc_regular - for regular tests of PSC functions
# opt_regular - regular optimizer tests for functions listed in optimizerTests list
# regular - regular tests for functions listed in allKernelFunctions list

num_internal_clusters = 20 #3
num_sparse_neighbors = 100 #10 #4
num_random = 15 #2
num_clusters = 20 #3
cluster_std_dev = 4 #1
num_samples = 500 #8
num_set = 20 #3
num_features = 500
metric = "euclidean"
#num_sparse_neighbors_full = num_samples #because less than this doesn't work for DisparitySum
num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
budget = 20
num_concepts = 50

# num_internal_clusters = 3 #3
# num_sparse_neighbors = 5 #10 #4
# num_random = 2 #2
# num_clusters = 3#3
# cluster_std_dev = 4 #1
# num_samples = 9
# num_set = 3 #3
# num_features = 2
# metric = "euclidean"
# #num_sparse_neighbors_full = num_samples #because less than this doesn't work for DisparitySum
# num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
# budget = 5
# num_concepts = 3

@pytest.fixture
def data():
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    # get num_set data points belonging to different clusters
    subset2 = []
    for i in range(num_set):
        #find the index of first point that belongs to cluster i
        diverse_index = cluster_ids.tolist().index(i)
        subset2.append(diverse_index)
    set2 = set(subset2[:-1])

    dataArray = np.array(data)
    return (num_samples, dataArray, set1, set2)

@pytest.fixture
def data_with_clusters():
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    # get num_set data points belonging to different clusters
    subset2 = []
    for i in range(num_set):
        #find the index of first point that belongs to cluster i
        diverse_index = cluster_ids.tolist().index(i)
        subset2.append(diverse_index)
    set2 = set(subset2[:-1])

    dataArray = np.array(data)
    return (num_samples, dataArray, set1, set2, cluster_ids)

@pytest.fixture
def data_features_log():
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    features = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    obj = FeatureBasedFunction(n=num_samples, features=features, numFeatures=num_features, sparse=False)

    return (obj, set1)

@pytest.fixture
def data_features_sqrt():
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    features = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    obj = FeatureBasedFunction(n=num_samples, features=features, numFeatures=num_features, mode=FeatureBased.squareRoot, sparse=False)

    return (obj, set1)

@pytest.fixture
def data_features_inverse():
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    features = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    obj = FeatureBasedFunction(n=num_samples, features=features, numFeatures=num_features, mode=FeatureBased.inverse, sparse=False)

    return (obj, set1)


@pytest.fixture
def data_concepts():
    cover_set = []
    np.random.seed(1)
    random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        cover_set.append(set(random.sample(list(range(num_concepts)), random.randint(0,num_concepts))))
    obj = SetCoverFunction(n=num_samples, cover_set=cover_set, num_concepts=num_concepts, concept_weights=concept_weights)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def data_prob_concepts():
    probs = []
    np.random.seed(1)
    random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        probs.append(np.random.rand(num_concepts).tolist())
    obj = ProbabilisticSetCoverFunction(n=num_samples, probs=probs, num_concepts=num_concepts, concept_weights=concept_weights)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def object_dense_cpp_kernel(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
    elif request.param == "DisparitySum":
        obj = DisparitySumFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
    elif request.param == "DisparityMin":
        obj = DisparityMinFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
    elif request.param == "GraphCut":
        obj = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, data=dataArray, metric=metric)
    elif request.param == "LogDeterminant":
        obj = LogDeterminantFunction(n=num_samples, mode="dense", lambdaVal=1, data=dataArray, metric=metric)
    else:
        return None
    return obj

@pytest.fixture
def object_dense_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    _, K_dense = create_kernel(dataArray, 'dense','euclidean')
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_master=False)
    elif request.param == "DisparitySum":
        obj = DisparitySumFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "DisparityMin":
        obj = DisparityMinFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "GraphCut":
        obj = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, ggsijs=K_dense)
    elif request.param == "LogDeterminant":
        obj = LogDeterminantFunction(n=num_samples, mode="dense", lambdaVal=1, data=dataArray, metric=metric)
    else:
        return None
    return obj

@pytest.fixture
def objects_dense_cpp_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    _, K_dense = create_kernel(dataArray, 'dense','euclidean')
    if request.param == "FacilityLocation":
        obj1 = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
        obj2 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_master=False)
    elif request.param == "DisparitySum":
        obj1 = DisparitySumFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
        obj2 = DisparitySumFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "DisparityMin":
        obj1 = DisparityMinFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
        obj2 = DisparityMinFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "GraphCut":
        obj1 = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, data=dataArray, metric=metric)
        obj2 = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, ggsijs=K_dense)
    elif request.param == "LogDeterminant":
        obj1 = LogDeterminantFunction(n=num_samples, mode="dense", data=dataArray, metric=metric, lambdaVal=1)
        obj2 = LogDeterminantFunction(n=num_samples, mode="dense", sijs = K_dense, lambdaVal=1)
    else:
        return None
    return obj1, obj2

@pytest.fixture
def object_sparse_cpp_kernel(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)
    elif request.param == "DisparitySum":
        obj = DisparitySumFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "DisparityMin":
        obj = DisparityMinFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "GraphCut":
        obj = GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)
    elif request.param == "LogDeterminant":
        obj = LogDeterminantFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full, lambdaVal=1)
    else:
        return None
    return obj

@pytest.fixture
def object_sparse_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    
    if request.param == "FacilityLocation":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors)
        obj = FacilityLocationFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "DisparitySum":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors_full)
        obj = DisparitySumFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "DisparityMin":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors_full)
        obj = DisparityMinFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "GraphCut":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors)
        obj = GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, ggsijs=K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "LogDeterminant":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors_full)
        obj = LogDeterminantFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full, lambdaVal=1)
    else:
        return None
    return obj

@pytest.fixture
def objects_sparse_cpp_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    
    if request.param == "FacilityLocation":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors)
        obj1 = FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)
        obj2 = FacilityLocationFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "DisparitySum":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors_full)
        obj1 = DisparitySumFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full)
        obj2 = DisparitySumFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "DisparityMin":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors_full)
        obj1 = DisparityMinFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full)
        obj2 = DisparityMinFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "GraphCut":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors)
        obj1= GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)
        obj2 = GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, ggsijs=K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "LogDeterminant":
        _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors_full)
        obj1 = LogDeterminantFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full, lambdaVal=1)
        obj2 = LogDeterminantFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full, lambdaVal=1)
    else:
        return None
    return obj1, obj2

@pytest.fixture
def object_clustered_mode_birch(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric=metric, num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_mode_user(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="clustered", cluster_labels=cluster_ids.tolist(), data=dataArray, metric=metric, num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_birch_multi(request, data):
    num_samples, dataArray, _, _ = data
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    return obj

@pytest.fixture
def object_clustered_user_multi(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    return obj

@pytest.fixture
def object_clustered_birch_single(request, data):
    num_samples, dataArray, _, _ = data
    obj = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    return obj

@pytest.fixture
def object_clustered_user_single(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    obj = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    return obj

@pytest.fixture
def objects_single_multi_user(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    obj1 = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    obj2 = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    return obj1, obj2

@pytest.fixture
def objects_mode_clustered_user(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    if request.param == "FacilityLocation":
        obj1 = FacilityLocationFunction(n=num_samples, mode="clustered", cluster_labels=cluster_ids.tolist(), data=dataArray, metric=metric, num_clusters=num_internal_clusters)
        obj2 = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    else:
        return None
    return obj1, obj2

@pytest.fixture
def objects_single_multi_birch(request, data):
    num_samples, dataArray, _, _ = data
    obj1 = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    obj2 = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    return obj1, obj2

@pytest.fixture
def objects_mode_clustered_birch(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj1 = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric=metric, num_clusters=num_internal_clusters)
        obj2 = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    else:
        return None
    return obj1, obj2


class TestAll:
    ############ 6 tests for dense cpp kernel #######################
    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", allKernelFunctions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_eval_groundset(self, object_dense_cpp_kernel):
        groundSet = object_dense_cpp_kernel.getEffectiveGroundSet()
        eval = object_dense_cpp_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", allKernelFunctions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_eval_emptyset(self, object_dense_cpp_kernel):
        eval = object_dense_cpp_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", allKernelFunctions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_gain_on_empty(self, data, object_dense_cpp_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_dense_cpp_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_dense_cpp_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_dense_cpp_kernel.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", allKernelFunctions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_eval_evalfast(self, data, object_dense_cpp_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_dense_cpp_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_dense_cpp_kernel.evaluate(subset)
        fastEval = object_dense_cpp_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", allKernelFunctions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_set_memoization(self, data, object_dense_cpp_kernel):
        _, _, set1, _ = data
        object_dense_cpp_kernel.setMemoization(set1)
        simpleEval = object_dense_cpp_kernel.evaluate(set1)
        fastEval = object_dense_cpp_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", allKernelFunctions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_gain(self, data, object_dense_cpp_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_dense_cpp_kernel.setMemoization(subset)
        firstEval = object_dense_cpp_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_dense_cpp_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_dense_cpp_kernel.marginalGain(subset, elem)
        fastGain = object_dense_cpp_kernel.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"

    ############ 6 tests for dense python kernel #######################

    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_py_kernel", allKernelFunctions, indirect=['object_dense_py_kernel'])
    def test_dense_py_eval_groundset(self, object_dense_py_kernel):
        groundSet = object_dense_py_kernel.getEffectiveGroundSet()
        eval = object_dense_py_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_py_kernel", allKernelFunctions, indirect=['object_dense_py_kernel'])
    def test_dense_py_eval_emptyset(self, object_dense_py_kernel):
        eval = object_dense_py_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_py_kernel", allKernelFunctions, indirect=['object_dense_py_kernel'])
    def test_dense_py_gain_on_empty(self, data, object_dense_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_dense_py_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_dense_py_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_dense_py_kernel.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_py_kernel", allKernelFunctions, indirect=['object_dense_py_kernel'])
    def test_dense_py_eval_evalfast(self, data, object_dense_py_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_dense_py_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_dense_py_kernel.evaluate(subset)
        fastEval = object_dense_py_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval,rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_py_kernel", allKernelFunctions, indirect=['object_dense_py_kernel'])
    def test_dense_py_set_memoization(self, data, object_dense_py_kernel):
        _, _, set1, _ = data
        object_dense_py_kernel.setMemoization(set1)
        simpleEval = object_dense_py_kernel.evaluate(set1)
        fastEval = object_dense_py_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_dense_py_kernel", allKernelFunctions, indirect=['object_dense_py_kernel'])
    def test_dense_py_gain(self, data, object_dense_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_dense_py_kernel.setMemoization(subset)
        firstEval = object_dense_py_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_dense_py_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_dense_py_kernel.marginalGain(subset, elem)
        fastGain = object_dense_py_kernel.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    

    ############ 6 tests for sparse cpp kernel #######################

    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_cpp_kernel", allKernelFunctions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_eval_groundset(self, object_sparse_cpp_kernel):
        groundSet = object_sparse_cpp_kernel.getEffectiveGroundSet()
        eval = object_sparse_cpp_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_cpp_kernel", allKernelFunctions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_eval_emptyset(self, object_sparse_cpp_kernel):
        eval = object_sparse_cpp_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_cpp_kernel", allKernelFunctions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_gain_on_empty(self, data, object_sparse_cpp_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_sparse_cpp_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_sparse_cpp_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_sparse_cpp_kernel.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_cpp_kernel", allKernelFunctions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_eval_evalfast(self, data, object_sparse_cpp_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_sparse_cpp_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_sparse_cpp_kernel.evaluate(subset)
        fastEval = object_sparse_cpp_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_cpp_kernel", allKernelFunctions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_set_memoization(self, data, object_sparse_cpp_kernel):
        _, _, set1, _ = data
        object_sparse_cpp_kernel.setMemoization(set1)
        simpleEval = object_sparse_cpp_kernel.evaluate(set1)
        fastEval = object_sparse_cpp_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_cpp_kernel", allKernelFunctions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_gain(self, data, object_sparse_cpp_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_sparse_cpp_kernel.setMemoization(subset)
        firstEval = object_sparse_cpp_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_sparse_cpp_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_sparse_cpp_kernel.marginalGain(subset, elem)
        fastGain = object_sparse_cpp_kernel.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ############ 6 tests for sparse python kernel #######################

    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_py_kernel", allKernelFunctions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_eval_groundset(self, object_sparse_py_kernel):
        groundSet = object_sparse_py_kernel.getEffectiveGroundSet()
        eval = object_sparse_py_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_py_kernel", allKernelFunctions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_eval_emptyset(self, object_sparse_py_kernel):
        eval = object_sparse_py_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_py_kernel", allKernelFunctions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_gain_on_empty(self, data, object_sparse_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_sparse_py_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_sparse_py_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_sparse_py_kernel.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_py_kernel", allKernelFunctions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_eval_evalfast(self, data, object_sparse_py_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_sparse_py_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_sparse_py_kernel.evaluate(subset)
        fastEval = object_sparse_py_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_py_kernel", allKernelFunctions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_set_memoization(self, data, object_sparse_py_kernel):
        _, _, set1, _ = data
        object_sparse_py_kernel.setMemoization(set1)
        simpleEval = object_sparse_py_kernel.evaluate(set1)
        fastEval = object_sparse_py_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_sparse_py_kernel", allKernelFunctions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_gain(self, data, object_sparse_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_sparse_py_kernel.setMemoization(subset)
        firstEval = object_sparse_py_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_sparse_py_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_sparse_py_kernel.marginalGain(subset, elem)
        fastGain = object_sparse_py_kernel.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"

    ############ 6 tests for clustered mode with internel clustering #######################

    @pytest.mark.parametrize("object_clustered_mode_birch", clusteredModeFunctions, indirect=['object_clustered_mode_birch'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_birch_eval_groundset(self, object_clustered_mode_birch):
        groundSet = object_clustered_mode_birch.getEffectiveGroundSet()
        eval = object_clustered_mode_birch.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.parametrize("object_clustered_mode_birch", clusteredModeFunctions, indirect=['object_clustered_mode_birch'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_birch_eval_emptyset(self, object_clustered_mode_birch):
        eval = object_clustered_mode_birch.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"

    @pytest.mark.parametrize("object_clustered_mode_birch", clusteredModeFunctions, indirect=['object_clustered_mode_birch'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_birch_gain_on_empty(self, data, object_clustered_mode_birch):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_clustered_mode_birch.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_clustered_mode_birch.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_clustered_mode_birch.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.parametrize("object_clustered_mode_birch", clusteredModeFunctions, indirect=['object_clustered_mode_birch'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_birch_eval_evalfast(self, data, object_clustered_mode_birch):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_mode_birch.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_mode_birch.evaluate(subset)
        fastEval = object_clustered_mode_birch.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_mode_birch", clusteredModeFunctions, indirect=['object_clustered_mode_birch'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_birch_set_memoization(self, data, object_clustered_mode_birch):
        _, _, set1, _ = data
        object_clustered_mode_birch.setMemoization(set1)
        simpleEval = object_clustered_mode_birch.evaluate(set1)
        fastEval = object_clustered_mode_birch.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_mode_birch", clusteredModeFunctions, indirect=['object_clustered_mode_birch'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_birch_gain(self, data, object_clustered_mode_birch):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_clustered_mode_birch.setMemoization(subset)
        firstEval = object_clustered_mode_birch.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_clustered_mode_birch.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_clustered_mode_birch.marginalGain(subset, elem)
        fastGain = object_clustered_mode_birch.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ############ 6 tests for clustered mode with user provided clustering #######################

    @pytest.mark.parametrize("object_clustered_mode_user", clusteredModeFunctions, indirect=['object_clustered_mode_user'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_user_eval_groundset(self, object_clustered_mode_user):
        groundSet = object_clustered_mode_user.getEffectiveGroundSet()
        eval = object_clustered_mode_user.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.parametrize("object_clustered_mode_user", clusteredModeFunctions, indirect=['object_clustered_mode_user'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_user_eval_emptyset(self, object_clustered_mode_user):
        eval = object_clustered_mode_user.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"

    @pytest.mark.parametrize("object_clustered_mode_user", clusteredModeFunctions, indirect=['object_clustered_mode_user'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_user_gain_on_empty(self, data, object_clustered_mode_user):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_clustered_mode_user.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_clustered_mode_user.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_clustered_mode_user.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.parametrize("object_clustered_mode_user", clusteredModeFunctions, indirect=['object_clustered_mode_user'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_user_eval_evalfast(self, data, object_clustered_mode_user):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_mode_user.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_mode_user.evaluate(subset)
        fastEval = object_clustered_mode_user.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_mode_user", clusteredModeFunctions, indirect=['object_clustered_mode_user'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_user_set_memoization(self, data, object_clustered_mode_user):
        _, _, set1, _ = data
        object_clustered_mode_user.setMemoization(set1)
        simpleEval = object_clustered_mode_user.evaluate(set1)
        fastEval = object_clustered_mode_user.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_mode_user", clusteredModeFunctions, indirect=['object_clustered_mode_user'])
    @pytest.mark.clustered_mode
    def test_clustered_mode_user_gain(self, data, object_clustered_mode_user):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_clustered_mode_user.setMemoization(subset)
        firstEval = object_clustered_mode_user.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_clustered_mode_user.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_clustered_mode_user.marginalGain(subset, elem)
        fastGain = object_clustered_mode_user.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"

    ############ 6 tests for clustered function with internel clustering and multiple small kernels #######################

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_multi", allKernelFunctions, indirect=['object_clustered_birch_multi'])
    def test_clustered_birch_multi_eval_groundset(self, object_clustered_birch_multi):
        groundSet = object_clustered_birch_multi.getEffectiveGroundSet()
        eval = object_clustered_birch_multi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_multi", allKernelFunctions, indirect=['object_clustered_birch_multi'])
    def test_clustered_birch_multi_eval_emptyset(self, object_clustered_birch_multi):
        eval = object_clustered_birch_multi.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_multi", allKernelFunctions, indirect=['object_clustered_birch_multi'])
    def test_clustered_birch_multi_gain_on_empty(self, data, object_clustered_birch_multi):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_clustered_birch_multi.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_clustered_birch_multi.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_clustered_birch_multi.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_multi", allKernelFunctions, indirect=['object_clustered_birch_multi'])
    def test_clustered_birch_multi_eval_evalfast(self, data, object_clustered_birch_multi):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_birch_multi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_birch_multi.evaluate(subset)
        fastEval = object_clustered_birch_multi.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_multi", allKernelFunctions, indirect=['object_clustered_birch_multi'])
    def test_clustered_birch_multi_set_memoization(self, data, object_clustered_birch_multi):
        _, _, set1, _ = data
        object_clustered_birch_multi.setMemoization(set1)
        simpleEval = object_clustered_birch_multi.evaluate(set1)
        fastEval = object_clustered_birch_multi.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_multi", allKernelFunctions, indirect=['object_clustered_birch_multi'])
    def test_clustered_birch_multi_gain(self, data, object_clustered_birch_multi):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_clustered_birch_multi.setMemoization(subset)
        firstEval = object_clustered_birch_multi.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_clustered_birch_multi.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_clustered_birch_multi.marginalGain(subset, elem)
        fastGain = object_clustered_birch_multi.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ############ 6 tests for clustered function with user provided clustering and multiple small kernels #######################

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_multi", allKernelFunctions, indirect=['object_clustered_user_multi'])
    def test_clustered_user_multi_eval_groundset(self, object_clustered_user_multi):
        groundSet = object_clustered_user_multi.getEffectiveGroundSet()
        eval = object_clustered_user_multi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_multi", allKernelFunctions, indirect=['object_clustered_user_multi'])
    def test_clustered_user_multi_eval_emptyset(self, object_clustered_user_multi):
        eval = object_clustered_user_multi.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_multi", allKernelFunctions, indirect=['object_clustered_user_multi'])
    def test_clustered_user_multi_gain_on_empty(self, data, object_clustered_user_multi):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_clustered_user_multi.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_clustered_user_multi.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_clustered_user_multi.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_multi", allKernelFunctions, indirect=['object_clustered_user_multi'])
    def test_clustered_user_multi_eval_evalfast(self, data, object_clustered_user_multi):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_user_multi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_user_multi.evaluate(subset)
        fastEval = object_clustered_user_multi.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval,fastEval,rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_multi", allKernelFunctions, indirect=['object_clustered_user_multi'])
    def test_clustered_user_multi_set_memoization(self, data, object_clustered_user_multi):
        _, _, set1, _ = data
        object_clustered_user_multi.setMemoization(set1)
        simpleEval = object_clustered_user_multi.evaluate(set1)
        fastEval = object_clustered_user_multi.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_multi", allKernelFunctions, indirect=['object_clustered_user_multi'])
    def test_clustered_user_multi_gain(self, data, object_clustered_user_multi):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_clustered_user_multi.setMemoization(subset)
        firstEval = object_clustered_user_multi.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_clustered_user_multi.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_clustered_user_multi.marginalGain(subset, elem)
        fastGain = object_clustered_user_multi.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"

    ############ 6 tests for clustered function with internel clustering and single kernel #######################

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_single", allKernelFunctions, indirect=['object_clustered_birch_single'])
    def test_clustered_birch_single_eval_groundset(self, object_clustered_birch_single):
        groundSet = object_clustered_birch_single.getEffectiveGroundSet()
        eval = object_clustered_birch_single.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_single", allKernelFunctions, indirect=['object_clustered_birch_single'])
    def test_clustered_birch_single_eval_emptyset(self, object_clustered_birch_single):
        eval = object_clustered_birch_single.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_single", allKernelFunctions, indirect=['object_clustered_birch_single'])
    def test_clustered_birch_single_gain_on_empty(self, data, object_clustered_birch_single):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_clustered_birch_single.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_clustered_birch_single.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_clustered_birch_single.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_single", allKernelFunctions, indirect=['object_clustered_birch_single'])
    def test_clustered_birch_single_eval_evalfast(self, data, object_clustered_birch_single):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_birch_single.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_birch_single.evaluate(subset)
        fastEval = object_clustered_birch_single.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_single", allKernelFunctions, indirect=['object_clustered_birch_single'])
    def test_clustered_birch_single_set_memoization(self, data, object_clustered_birch_single):
        _, _, set1, _ = data
        object_clustered_birch_single.setMemoization(set1)
        simpleEval = object_clustered_birch_single.evaluate(set1)
        fastEval = object_clustered_birch_single.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_single", allKernelFunctions, indirect=['object_clustered_birch_single'])
    def test_clustered_birch_single_set_memoization_groundset(self, data, object_clustered_birch_single):
        groundSet = object_clustered_birch_single.getEffectiveGroundSet()
        object_clustered_birch_single.setMemoization(groundSet)
        simpleEval = object_clustered_birch_single.evaluate(groundSet)
        fastEval = object_clustered_birch_single.evaluateWithMemoization(groundSet)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization for groundSet"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_birch_single", allKernelFunctions, indirect=['object_clustered_birch_single'])
    def test_clustered_birch_single_gain(self, data, object_clustered_birch_single):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_clustered_birch_single.setMemoization(subset)
        firstEval = object_clustered_birch_single.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_clustered_birch_single.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_clustered_birch_single.marginalGain(subset, elem)
        fastGain = object_clustered_birch_single.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ############ 6 tests for clustered function with user provided clustering and single kernel #######################

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_single", allKernelFunctions, indirect=['object_clustered_user_single'])
    def test_clustered_user_single_eval_groundset(self, object_clustered_user_single):
        groundSet = object_clustered_user_single.getEffectiveGroundSet()
        eval = object_clustered_user_single.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_single", allKernelFunctions, indirect=['object_clustered_user_single'])
    def test_clustered_user_single_eval_emptyset(self, object_clustered_user_single):
        eval = object_clustered_user_single.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_single", allKernelFunctions, indirect=['object_clustered_user_single'])
    def test_clustered_user_single_gain_on_empty(self, data, object_clustered_user_single):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_clustered_user_single.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_clustered_user_single.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_clustered_user_single.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_single", allKernelFunctions, indirect=['object_clustered_user_single'])
    def test_clustered_user_single_eval_evalfast(self, data, object_clustered_user_single):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_user_single.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_user_single.evaluate(subset)
        fastEval = object_clustered_user_single.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_single", allKernelFunctions, indirect=['object_clustered_user_single'])
    def test_clustered_user_single_set_memoization(self, data, object_clustered_user_single):
        _, _, set1, _ = data
        object_clustered_user_single.setMemoization(set1)
        simpleEval = object_clustered_user_single.evaluate(set1)
        fastEval = object_clustered_user_single.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.regular
    @pytest.mark.parametrize("object_clustered_user_single", allKernelFunctions, indirect=['object_clustered_user_single'])
    def test_clustered_user_single_gain(self, data, object_clustered_user_single):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_clustered_user_single.setMemoization(subset)
        firstEval = object_clustered_user_single.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_clustered_user_single.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_clustered_user_single.marginalGain(subset, elem)
        fastGain = object_clustered_user_single.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"

    ######### 4 Tests to check the consistency between CPP dense and Py dense

    @pytest.mark.regular
    @pytest.mark.parametrize("objects_dense_cpp_py_kernel", allKernelFunctions, indirect=['objects_dense_cpp_py_kernel'])
    def test_dense_cpp_py_eval(self, data, objects_dense_cpp_py_kernel):
        _, _, set1, _ = data
        eval1 = objects_dense_cpp_py_kernel[0].evaluate(set1)
        eval2 = objects_dense_cpp_py_kernel[1].evaluate(set1)
        assert math.isclose(eval1, eval2, rel_tol=1e-05), "Mismatch between evaluate() of CPP dense and PY dense"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_dense_cpp_py_kernel", allKernelFunctions, indirect=['objects_dense_cpp_py_kernel'])
    def test_dense_cpp_py_gain(self, data, objects_dense_cpp_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        gain1 = objects_dense_cpp_py_kernel[0].marginalGain(subset, elem)
        gain2 = objects_dense_cpp_py_kernel[1].marginalGain(subset, elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch between marginalGain() of CPP dense and PY dense"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_dense_cpp_py_kernel", allKernelFunctions, indirect=['objects_dense_cpp_py_kernel'])
    def test_dense_cpp_py_evalFast(self, data, objects_dense_cpp_py_kernel):
        _, _, set1, _ = data
        objects_dense_cpp_py_kernel[0].setMemoization(set1)
        objects_dense_cpp_py_kernel[1].setMemoization(set1)
        evalFast1 = objects_dense_cpp_py_kernel[0].evaluateWithMemoization(set1)
        evalFast2 = objects_dense_cpp_py_kernel[1].evaluateWithMemoization(set1)
        assert math.isclose(evalFast1, evalFast2, rel_tol=1e-05), "Mismatch between evaluateWithMemoization() of CPP dense and PY dense"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_dense_cpp_py_kernel", allKernelFunctions, indirect=['objects_dense_cpp_py_kernel'])
    def test_dense_cpp_py_gainFast(self, data, objects_dense_cpp_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        objects_dense_cpp_py_kernel[0].setMemoization(subset)
        objects_dense_cpp_py_kernel[1].setMemoization(subset)
        gainFast1 = objects_dense_cpp_py_kernel[0].marginalGainWithMemoization(subset, elem)
        gainFast2 = objects_dense_cpp_py_kernel[1].marginalGainWithMemoization(subset, elem)
        assert math.isclose(gainFast1, gainFast2, rel_tol=1e-05), "Mismatch between marginalGainWithMemoization() of CPP dense and PY dense"

    ######### 4 Tests to check the consistency between CPP sparse and Py sparse

    @pytest.mark.regular
    @pytest.mark.parametrize("objects_sparse_cpp_py_kernel", allKernelFunctions, indirect=['objects_sparse_cpp_py_kernel'])
    def test_sparse_cpp_py_eval(self, data, objects_sparse_cpp_py_kernel):
        _, _, set1, _ = data
        eval1 = objects_sparse_cpp_py_kernel[0].evaluate(set1)
        eval2 = objects_sparse_cpp_py_kernel[1].evaluate(set1)
        assert math.isclose(eval1, eval2, rel_tol=1e-05), "Mismatch between evaluate() of CPP sparse and PY sparse"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_sparse_cpp_py_kernel", allKernelFunctions, indirect=['objects_sparse_cpp_py_kernel'])
    def test_sparse_cpp_py_gain(self, data, objects_sparse_cpp_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        gain1 = objects_sparse_cpp_py_kernel[0].marginalGain(subset, elem)
        gain2 = objects_sparse_cpp_py_kernel[1].marginalGain(subset, elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch between marginalGain() of CPP sparse and PY sparse"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_sparse_cpp_py_kernel", allKernelFunctions, indirect=['objects_sparse_cpp_py_kernel'])
    def test_sparse_cpp_py_evalFast(self, data, objects_sparse_cpp_py_kernel):
        _, _, set1, _ = data
        objects_sparse_cpp_py_kernel[0].setMemoization(set1)
        objects_sparse_cpp_py_kernel[1].setMemoization(set1)
        evalFast1 = objects_sparse_cpp_py_kernel[0].evaluateWithMemoization(set1)
        evalFast2 = objects_sparse_cpp_py_kernel[1].evaluateWithMemoization(set1)
        assert math.isclose(evalFast1, evalFast2, rel_tol=1e-05), "Mismatch between evaluateWithMemoization() of CPP sparse and PY sparse"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_sparse_cpp_py_kernel", allKernelFunctions, indirect=['objects_sparse_cpp_py_kernel'])
    def test_sparse_cpp_py_gainFast(self, data, objects_sparse_cpp_py_kernel):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        objects_sparse_cpp_py_kernel[0].setMemoization(subset)
        objects_sparse_cpp_py_kernel[1].setMemoization(subset)
        gainFast1 = objects_sparse_cpp_py_kernel[0].marginalGainWithMemoization(subset, elem)
        gainFast2 = objects_sparse_cpp_py_kernel[1].marginalGainWithMemoization(subset, elem)
        assert math.isclose(gainFast1, gainFast2, rel_tol=1e-05), "Mismatch between marginalGainWithMemoization() of CPP sparse and PY sparse"

    ######### Tests to check the consistency between clustered mode and clustered function multi when user provides the clusters

    @pytest.mark.parametrize("objects_mode_clustered_user", clusteredModeFunctions, indirect=['objects_mode_clustered_user'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_user_eval(self, data, objects_mode_clustered_user):
        _, _, set1, _ = data
        eval1 = objects_mode_clustered_user[0].evaluate(set1)
        eval2 = objects_mode_clustered_user[1].evaluate(set1)
        assert math.isclose(eval1, eval2, rel_tol=1e-05), "Mismatch between evaluate() of clustered mode and clustered function multi"
    
    @pytest.mark.parametrize("objects_mode_clustered_user", clusteredModeFunctions, indirect=['objects_mode_clustered_user'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_user_gain(self, data, objects_mode_clustered_user):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        gain1 = objects_mode_clustered_user[0].marginalGain(subset, elem)
        gain2 = objects_mode_clustered_user[1].marginalGain(subset, elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch between marginalGain() of clustered mode and clustered function multi"
    
    @pytest.mark.parametrize("objects_mode_clustered_user", clusteredModeFunctions, indirect=['objects_mode_clustered_user'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_user_evalFast(self, data, objects_mode_clustered_user):
        _, _, set1, _ = data
        objects_mode_clustered_user[0].setMemoization(set1)
        objects_mode_clustered_user[1].setMemoization(set1)
        evalFast1 = objects_mode_clustered_user[0].evaluateWithMemoization(set1)
        evalFast2 = objects_mode_clustered_user[1].evaluateWithMemoization(set1)
        assert math.isclose(evalFast1, evalFast2, rel_tol=1e-05), "Mismatch between evaluateWithMemoization() of clustered mode and clustered function multi"
    
    @pytest.mark.parametrize("objects_mode_clustered_user", clusteredModeFunctions, indirect=['objects_mode_clustered_user'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_user_gainFast(self, data, objects_mode_clustered_user):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        objects_mode_clustered_user[0].setMemoization(subset)
        objects_mode_clustered_user[1].setMemoization(subset)
        gainFast1 = objects_mode_clustered_user[0].marginalGainWithMemoization(subset, elem)
        gainFast2 = objects_mode_clustered_user[1].marginalGainWithMemoization(subset, elem)
        assert math.isclose(gainFast1, gainFast2, rel_tol=1e-05), "Mismatch between marginalGainWithMemoization() of clustered mode and clustered function multi"
    
    ######### Tests to check the consistency between clustered function single and multi when user provides the clusters

    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_user", allKernelFunctions, indirect=['objects_single_multi_user'])
    def test_objects_single_multi_user_eval(self, data, objects_single_multi_user):
        _, _, set1, _ = data
        eval1 = objects_single_multi_user[0].evaluate(set1)
        eval2 = objects_single_multi_user[1].evaluate(set1)
        assert math.isclose(eval1, eval2, rel_tol=1e-05), "Mismatch between evaluate() of clustered function single and clustered function multi"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_user", allKernelFunctions, indirect=['objects_single_multi_user'])
    def test_objects_single_multi_user_gain(self, data, objects_single_multi_user):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        gain1 = objects_single_multi_user[0].marginalGain(subset, elem)
        gain2 = objects_single_multi_user[1].marginalGain(subset, elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch between marginalGain() of clustered function single and clustered function multi"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_user", allKernelFunctions, indirect=['objects_single_multi_user'])
    def test_objects_single_multi_user_evalFast(self, data, objects_single_multi_user):
        _, _, set1, _ = data
        objects_single_multi_user[0].setMemoization(set1)
        objects_single_multi_user[1].setMemoization(set1)
        evalFast1 = objects_single_multi_user[0].evaluateWithMemoization(set1)
        evalFast2 = objects_single_multi_user[1].evaluateWithMemoization(set1)
        assert math.isclose(evalFast1, evalFast2, rel_tol=1e-05), "Mismatch between evaluateWithMemoization() of clustered function single and clustered function multi"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_user", allKernelFunctions, indirect=['objects_single_multi_user'])
    def test_objects_single_multi_user_gainFast(self, data, objects_single_multi_user):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        objects_single_multi_user[0].setMemoization(subset)
        objects_single_multi_user[1].setMemoization(subset)
        gainFast1 = objects_single_multi_user[0].marginalGainWithMemoization(subset, elem)
        gainFast2 = objects_single_multi_user[1].marginalGainWithMemoization(subset, elem)
        assert math.isclose(gainFast1, gainFast2, rel_tol=1e-05), "Mismatch between marginalGainWithMemoization() of clustered function single and clustered function multi"
    
    ######### Tests to check the consistency between clustered mode and clustered function multi when internal BIRCH clustering is used

    @pytest.mark.parametrize("objects_mode_clustered_birch", clusteredModeFunctions, indirect=['objects_mode_clustered_birch'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_birch_eval(self, data, objects_mode_clustered_birch):
        _, _, set1, _ = data
        eval1 = objects_mode_clustered_birch[0].evaluate(set1)
        eval2 = objects_mode_clustered_birch[1].evaluate(set1)
        assert math.isclose(eval1, eval2, rel_tol=1e-05), "Mismatch between evaluate() of clustered mode and clustered function multi"
    
    @pytest.mark.parametrize("objects_mode_clustered_birch", clusteredModeFunctions, indirect=['objects_mode_clustered_birch'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_birch_gain(self, data, objects_mode_clustered_birch):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        gain1 = objects_mode_clustered_birch[0].marginalGain(subset, elem)
        gain2 = objects_mode_clustered_birch[1].marginalGain(subset, elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch between marginalGain() of clustered mode and clustered function multi"
    
    @pytest.mark.parametrize("objects_mode_clustered_birch", clusteredModeFunctions, indirect=['objects_mode_clustered_birch'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_birch_evalFast(self, data, objects_mode_clustered_birch):
        _, _, set1, _ = data
        objects_mode_clustered_birch[0].setMemoization(set1)
        objects_mode_clustered_birch[1].setMemoization(set1)
        evalFast1 = objects_mode_clustered_birch[0].evaluateWithMemoization(set1)
        evalFast2 = objects_mode_clustered_birch[1].evaluateWithMemoization(set1)
        assert math.isclose(evalFast1, evalFast2, rel_tol=1e-05), "Mismatch between evaluateWithMemoization() of clustered mode and clustered function multi"
    
    @pytest.mark.parametrize("objects_mode_clustered_birch", clusteredModeFunctions, indirect=['objects_mode_clustered_birch'])
    @pytest.mark.clustered_mode
    def test_objects_mode_clustered_birch_gainFast(self, data, objects_mode_clustered_birch):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        objects_mode_clustered_birch[0].setMemoization(subset)
        objects_mode_clustered_birch[1].setMemoization(subset)
        gainFast1 = objects_mode_clustered_birch[0].marginalGainWithMemoization(subset, elem)
        gainFast2 = objects_mode_clustered_birch[1].marginalGainWithMemoization(subset, elem)
        assert math.isclose(gainFast1, gainFast2, rel_tol=1e-05), "Mismatch between marginalGainWithMemoization() of clustered mode and clustered function multi"

    ######### Tests to check the consistency between clustered single and clustered multi when internal BIRCH clustering is used

    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_birch", allKernelFunctions, indirect=['objects_single_multi_birch'])
    def test_objects_single_multi_birch_eval(self, data, objects_single_multi_birch):
        _, _, set1, _ = data
        eval1 = objects_single_multi_birch[0].evaluate(set1)
        eval2 = objects_single_multi_birch[1].evaluate(set1)
        assert math.isclose(eval1, eval2, rel_tol=1e-05), "Mismatch between evaluate() of clustered function single and clustered function multi"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_birch", allKernelFunctions, indirect=['objects_single_multi_birch'])
    def test_objects_single_multi_birch_gain(self, data, objects_single_multi_birch):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        gain1 = objects_single_multi_birch[0].marginalGain(subset, elem)
        gain2 = objects_single_multi_birch[1].marginalGain(subset, elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch between marginalGain() of clustered function single and clustered function multi"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_birch", allKernelFunctions, indirect=['objects_single_multi_birch'])
    def test_objects_single_multi_birch_evalFast(self, data, objects_single_multi_birch):
        _, _, set1, _ = data
        objects_single_multi_birch[0].setMemoization(set1)
        objects_single_multi_birch[1].setMemoization(set1)
        evalFast1 = objects_single_multi_birch[0].evaluateWithMemoization(set1)
        evalFast2 = objects_single_multi_birch[1].evaluateWithMemoization(set1)
        assert math.isclose(evalFast1, evalFast2, rel_tol=1e-05), "Mismatch between evaluateWithMemoization() of clustered function single and clustered function multi"
    
    @pytest.mark.regular
    @pytest.mark.parametrize("objects_single_multi_birch", allKernelFunctions, indirect=['objects_single_multi_birch'])
    def test_objects_single_multi_birch_gainFast(self, data, objects_single_multi_birch):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        objects_single_multi_birch[0].setMemoization(subset)
        objects_single_multi_birch[1].setMemoization(subset)
        gainFast1 = objects_single_multi_birch[0].marginalGainWithMemoization(subset, elem)
        gainFast2 = objects_single_multi_birch[1].marginalGainWithMemoization(subset, elem)
        assert math.isclose(gainFast1, gainFast2, rel_tol=1e-05), "Mismatch between marginalGainWithMemoization() of clustered function single and clustered function multi"

    ######## Tests for optimizers ################
    @pytest.mark.opt_regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", optimizerTests, indirect=['object_dense_cpp_kernel'])
    def test_naive_lazy(self, object_dense_cpp_kernel):
        greedyListNaive = object_dense_cpp_kernel.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_dense_cpp_kernel.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListNaive == greedyListLazy, "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.opt_regular
    @pytest.mark.parametrize("object_dense_cpp_kernel", optimizerTests, indirect=['object_dense_cpp_kernel'])
    def test_stochastic_lazierThanLazy(self, object_dense_cpp_kernel):
        greedyListStochastic = object_dense_cpp_kernel.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_dense_cpp_kernel.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListStochastic == greedyListLazierThanLazy, "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ######## Optimizers test for FeatureBased Logarithmic #####################
    @pytest.mark.fb_opt
    def test_fb_log_optimizer_naive_lazy(self, data_features_log):
        object_fb, _ = data_features_log
        greedyListNaive = object_fb.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_fb.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListNaive == greedyListLazy, "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.fb_opt
    def test_fb_log_optimizer_stochastic_lazierThanLazy(self, data_features_log):
        object_fb, _ = data_features_log
        greedyListStochastic = object_fb.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_fb.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListStochastic == greedyListLazierThanLazy, "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for FeatureBased Logarithmic Function #######################
    @pytest.mark.fb_regular
    def test_fb_log_eval_groundset(self, data_features_log):
        object_fb, _ = data_features_log
        groundSet = object_fb.getEffectiveGroundSet()
        eval = object_fb.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.fb_regular
    def test_fb_log_eval_emptyset(self, data_features_log):
        object_fb, _ = data_features_log
        eval = object_fb.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.fb_regular
    def test_fb_log_gain_on_empty(self, data_features_log):
        object_fb, set1 = data_features_log
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_fb.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_fb.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_fb.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.fb_regular
    def test_fb_log_eval_evalfast(self, data_features_log):
        object_fb, set1 = data_features_log
        subset = set()
        for elem in set1:
            object_fb.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_fb.evaluate(subset)
        fastEval = object_fb.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.fb_regular
    def test_fb_log_set_memoization(self, data_features_log):
        object_fb, set1 = data_features_log
        object_fb.setMemoization(set1)
        simpleEval = object_fb.evaluate(set1)
        fastEval = object_fb.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.fb_regular
    def test_fb_log_gain(self, data_features_log):
        object_fb, set1 = data_features_log
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_fb.setMemoization(subset)
        firstEval = object_fb.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_fb.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_fb.marginalGain(subset, elem)
        fastGain = object_fb.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ######## Optimizers test for FeatureBased SquareRoot #####################
    @pytest.mark.fb_opt
    def test_fb_sqrt_optimizer_naive_lazy(self, data_features_sqrt):
        object_fb, _ = data_features_sqrt
        greedyListNaive = object_fb.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_fb.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListNaive == greedyListLazy, "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.fb_opt
    def test_fb_sqrt_optimizer_stochastic_lazierThanLazy(self, data_features_sqrt):
        object_fb, _ = data_features_sqrt
        greedyListStochastic = object_fb.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_fb.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListStochastic == greedyListLazierThanLazy, "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for FeatureBased SquareRoot Function #######################
    @pytest.mark.fb_regular
    def test_fb_sqrt_eval_groundset(self, data_features_sqrt):
        object_fb, _ = data_features_sqrt
        groundSet = object_fb.getEffectiveGroundSet()
        eval = object_fb.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.fb_regular
    def test_fb_sqrt_eval_emptyset(self, data_features_sqrt):
        object_fb, _ = data_features_sqrt
        eval = object_fb.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.fb_regular
    def test_fb_sqrt_gain_on_empty(self, data_features_sqrt):
        object_fb, set1 = data_features_sqrt
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_fb.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_fb.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_fb.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.fb_regular
    def test_fb_sqrt_eval_evalfast(self, data_features_sqrt):
        object_fb, set1 = data_features_sqrt
        subset = set()
        for elem in set1:
            object_fb.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_fb.evaluate(subset)
        fastEval = object_fb.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.fb_regular
    def test_fb_sqrt_set_memoization(self, data_features_sqrt):
        object_fb, set1 = data_features_sqrt
        object_fb.setMemoization(set1)
        simpleEval = object_fb.evaluate(set1)
        fastEval = object_fb.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.fb_regular
    def test_fb_sqrt_gain(self, data_features_sqrt):
        object_fb, set1 = data_features_sqrt
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_fb.setMemoization(subset)
        firstEval = object_fb.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_fb.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_fb.marginalGain(subset, elem)
        fastGain = object_fb.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ######## Optimizers test for FeatureBased Inverse #####################
    @pytest.mark.fb_opt
    def test_fb_inverse_optimizer_naive_lazy(self, data_features_inverse):
        object_fb, _ = data_features_inverse
        greedyListNaive = object_fb.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_fb.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListNaive == greedyListLazy, "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.fb_opt
    def test_fb_inverse_optimizer_stochastic_lazierThanLazy(self, data_features_inverse):
        object_fb, _ = data_features_inverse
        greedyListStochastic = object_fb.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_fb.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        assert greedyListStochastic == greedyListLazierThanLazy, "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for FeatureBased Inverse Function #######################
    @pytest.mark.fb_regular
    def test_fb_inverse_eval_groundset(self, data_features_inverse):
        object_fb, _ = data_features_inverse
        groundSet = object_fb.getEffectiveGroundSet()
        eval = object_fb.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.fb_regular
    def test_fb_inverse_eval_emptyset(self, data_features_inverse):
        object_fb, _ = data_features_inverse
        eval = object_fb.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.fb_regular
    def test_fb_inverse_gain_on_empty(self, data_features_inverse):
        object_fb, set1 = data_features_inverse
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_fb.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_fb.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_fb.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.fb_regular
    def test_fb_inverse_eval_evalfast(self, data_features_inverse):
        object_fb, set1 = data_features_inverse
        subset = set()
        for elem in set1:
            object_fb.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_fb.evaluate(subset)
        fastEval = object_fb.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.fb_regular
    def test_fb_inverse_set_memoization(self, data_features_inverse):
        object_fb, set1 = data_features_inverse
        object_fb.setMemoization(set1)
        simpleEval = object_fb.evaluate(set1)
        fastEval = object_fb.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.fb_regular
    def test_fb_inverse_gain(self, data_features_inverse):
        object_fb, set1 = data_features_inverse
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_fb.setMemoization(subset)
        firstEval = object_fb.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_fb.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_fb.marginalGain(subset, elem)
        fastGain = object_fb.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ######## Optimizers test for SetCover #####################
    @pytest.mark.sc_opt
    def test_sc_optimizer_naive_lazy(self, data_concepts):
        object_sc, _ = data_concepts
        greedyListNaive = object_sc.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_sc.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.sc_opt
    def test_sc_optimizer_stochastic_lazierThanLazy(self, data_concepts):
        object_sc, _ = data_concepts
        greedyListStochastic = object_sc.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_sc.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == lazierThanLazyGains, "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for SetCover Function #######################
    @pytest.mark.sc_regular
    def test_sc_eval_groundset(self, data_concepts):
        object_sc, _ = data_concepts
        groundSet = object_sc.getEffectiveGroundSet()
        eval = object_sc.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.sc_regular
    def test_sc_eval_emptyset(self, data_concepts):
        object_sc, _ = data_concepts
        eval = object_sc.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.sc_regular
    def test_sc_gain_on_empty(self, data_concepts):
        object_sc, set1 = data_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_sc.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_sc.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_sc.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.sc_regular
    def test_sc_eval_evalfast(self, data_concepts):
        object_sc, set1 = data_concepts
        subset = set()
        for elem in set1:
            object_sc.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_sc.evaluate(subset)
        fastEval = object_sc.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.sc_regular
    def test_sc_set_memoization(self, data_concepts):
        object_sc, set1 = data_concepts
        object_sc.setMemoization(set1)
        simpleEval = object_sc.evaluate(set1)
        fastEval = object_sc.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.sc_regular
    def test_sc_gain(self, data_concepts):
        object_sc, set1 = data_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_sc.setMemoization(subset)
        firstEval = object_sc.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_sc.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_sc.marginalGain(subset, elem)
        fastGain = object_sc.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
    
    ######## Optimizers test for ProbabilisticSetCover #####################
    @pytest.mark.psc_opt
    def test_psc_optimizer_naive_lazy(self, data_prob_concepts):
        object_psc, _ = data_prob_concepts
        greedyListNaive = object_psc.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_psc.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.psc_opt
    def test_psc_optimizer_stochastic_lazierThanLazy(self, data_prob_concepts):
        object_psc, _ = data_prob_concepts
        greedyListStochastic = object_psc.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_psc.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == lazierThanLazyGains, "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for ProbabilisticSetCover Function #######################
    @pytest.mark.psc_regular
    def test_psc_eval_groundset(self, data_prob_concepts):
        object_psc, _ = data_prob_concepts
        groundSet = object_psc.getEffectiveGroundSet()
        eval = object_psc.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.psc_regular
    def test_psc_eval_emptyset(self, data_prob_concepts):
        object_psc, _ = data_prob_concepts
        eval = object_psc.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.psc_regular
    def test_psc_gain_on_empty(self, data_prob_concepts):
        object_psc, set1 = data_prob_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_psc.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_psc.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_psc.marginalGain(set(), elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch for gain on empty set"
        #assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.psc_regular
    def test_psc_eval_evalfast(self, data_prob_concepts):
        object_psc, set1 = data_prob_concepts
        subset = set()
        for elem in set1:
            object_psc.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_psc.evaluate(subset)
        fastEval = object_psc.evaluateWithMemoization(subset)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.psc_regular
    def test_psc_set_memoization(self, data_prob_concepts):
        object_psc, set1 = data_prob_concepts
        object_psc.setMemoization(set1)
        simpleEval = object_psc.evaluate(set1)
        fastEval = object_psc.evaluateWithMemoization(set1)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.psc_regular
    def test_psc_gain(self, data_prob_concepts):
        object_psc, set1 = data_prob_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_psc.setMemoization(subset)
        firstEval = object_psc.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_psc.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_psc.marginalGain(subset, elem)
        fastGain = object_psc.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
        #assert naiveGain == simpleGain and simpleGain == fastGain, "Mismatch between naive, simple and fast margins"