import pytest
import math
from sklearn.datasets import make_blobs
import numpy as np
import random
from submodlib import FacilityLocationFunction
from submodlib import ClusteredFunction
from submodlib.helper import create_kernel

functions = ["FacilityLocation"]
num_internal_clusters = 10 #3
num_sparse_neighbors = 10 #4
num_random = 4 #2

@pytest.fixture
def data():
    num_clusters = 10 #3
    cluster_std_dev = 4 #1
    num_samples = 500 #8
    num_set = 6 #3
    budget = 10 #4

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
    data = list(map(tuple, points))
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    subset1xs = [xs[x] for x in subset1]
    subset1ys = [ys[x] for x in subset1]
    set1 = set(subset1[:-1])

    # get num_set data points belonging to different clusters
    subset2 = []
    for i in range(num_set):
        #find the index of first point that belongs to cluster i
        diverse_index = cluster_ids.tolist().index(i)
        subset2.append(diverse_index)
    subset2xs = [xs[x] for x in subset2]
    subset2ys = [ys[x] for x in subset2]
    set2 = set(subset2[:-1])

    dataArray = np.array(data)
    return (num_samples, dataArray, set1, set2)

@pytest.fixture
def data_with_clusters():
    num_clusters = 10 #3
    cluster_std_dev = 4 #1
    num_samples = 500 #8
    num_set = 6 #3
    budget = 10 #4

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
    data = list(map(tuple, points))
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]

    # get num_set data points belonging to cluster#1
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    subset1xs = [xs[x] for x in subset1]
    subset1ys = [ys[x] for x in subset1]
    set1 = set(subset1[:-1])

    # get num_set data points belonging to different clusters
    subset2 = []
    for i in range(num_set):
        #find the index of first point that belongs to cluster i
        diverse_index = cluster_ids.tolist().index(i)
        subset2.append(diverse_index)
    subset2xs = [xs[x] for x in subset2]
    subset2ys = [ys[x] for x in subset2]
    set2 = set(subset2[:-1])

    dataArray = np.array(data)
    return (num_samples, dataArray, set1, set2, cluster_ids)

@pytest.fixture
def object_dense_cpp_kernel(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
    else:
        return None
    return obj

@pytest.fixture
def object_dense_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    _, K_dense = create_kernel(dataArray, 'dense','euclidean')
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_master=False)
    else:
        return None
    return obj

@pytest.fixture
def object_sparse_cpp_kernel(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric="euclidean", num_neighbors=num_sparse_neighbors)
    else:
        return None
    return obj

@pytest.fixture
def object_sparse_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    _, K_sparse = create_kernel(dataArray, 'sparse','euclidean', num_neigh=num_sparse_neighbors)
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_mode_birch(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric="euclidean", num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_mode_user(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="clustered", cluster_labels=cluster_ids.tolist(), data=dataArray, metric="euclidean", num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_birch_multi(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_user_multi(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    if request.param == "FacilityLocation":
        obj = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_birch_single(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = ClusteredFunction(n=num_samples, mode="single", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_user_single(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    if request.param == "FacilityLocation":
        obj = ClusteredFunction(n=num_samples, mode="single", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    else:
        return None
    return obj


class TestAll:
    ############ 4 tests for dense cpp kernel #######################
    @pytest.mark.parametrize("object_dense_cpp_kernel", functions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_eval_groundset(self, object_dense_cpp_kernel):
        groundSet = object_dense_cpp_kernel.getEffectiveGroundSet()
        eval = object_dense_cpp_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_dense_cpp_kernel", functions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_eval_evalfast(self, data, object_dense_cpp_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_dense_cpp_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_dense_cpp_kernel.evaluate(subset)
        fastEval = object_dense_cpp_kernel.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_dense_cpp_kernel", functions, indirect=['object_dense_cpp_kernel'])
    def test_dense_cpp_set_memoization(self, data, object_dense_cpp_kernel):
        _, _, set1, _ = data
        object_dense_cpp_kernel.setMemoization(set1)
        simpleEval = object_dense_cpp_kernel.evaluate(set1)
        fastEval = object_dense_cpp_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_dense_cpp_kernel", functions, indirect=['object_dense_cpp_kernel'])
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

    ############ 4 tests for dense python kernel #######################

    @pytest.mark.parametrize("object_dense_py_kernel", functions, indirect=['object_dense_py_kernel'])
    def test_dense_py_eval_groundset(self, object_dense_py_kernel):
        groundSet = object_dense_py_kernel.getEffectiveGroundSet()
        eval = object_dense_py_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_dense_py_kernel", functions, indirect=['object_dense_py_kernel'])
    def test_dense_py_eval_evalfast(self, data, object_dense_py_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_dense_py_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_dense_py_kernel.evaluate(subset)
        fastEval = object_dense_py_kernel.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_dense_py_kernel", functions, indirect=['object_dense_py_kernel'])
    def test_dense_py_set_memoization(self, data, object_dense_py_kernel):
        _, _, set1, _ = data
        object_dense_py_kernel.setMemoization(set1)
        simpleEval = object_dense_py_kernel.evaluate(set1)
        fastEval = object_dense_py_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_dense_py_kernel", functions, indirect=['object_dense_py_kernel'])
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
    

    ############ 4 tests for sparse cpp kernel #######################

    @pytest.mark.parametrize("object_sparse_cpp_kernel", functions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_eval_groundset(self, object_sparse_cpp_kernel):
        groundSet = object_sparse_cpp_kernel.getEffectiveGroundSet()
        eval = object_sparse_cpp_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_sparse_cpp_kernel", functions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_eval_evalfast(self, data, object_sparse_cpp_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_sparse_cpp_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_sparse_cpp_kernel.evaluate(subset)
        fastEval = object_sparse_cpp_kernel.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_sparse_cpp_kernel", functions, indirect=['object_sparse_cpp_kernel'])
    def test_sparse_cpp_set_memoization(self, data, object_sparse_cpp_kernel):
        _, _, set1, _ = data
        object_sparse_cpp_kernel.setMemoization(set1)
        simpleEval = object_sparse_cpp_kernel.evaluate(set1)
        fastEval = object_sparse_cpp_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_sparse_cpp_kernel", functions, indirect=['object_sparse_cpp_kernel'])
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
    
    ############ 4 tests for sparse python kernel #######################

    @pytest.mark.parametrize("object_sparse_py_kernel", functions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_eval_groundset(self, object_sparse_py_kernel):
        groundSet = object_sparse_py_kernel.getEffectiveGroundSet()
        eval = object_sparse_py_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_sparse_py_kernel", functions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_eval_evalfast(self, data, object_sparse_py_kernel):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_sparse_py_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_sparse_py_kernel.evaluate(subset)
        fastEval = object_sparse_py_kernel.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_sparse_py_kernel", functions, indirect=['object_sparse_py_kernel'])
    def test_sparse_py_set_memoization(self, data, object_sparse_py_kernel):
        _, _, set1, _ = data
        object_sparse_py_kernel.setMemoization(set1)
        simpleEval = object_sparse_py_kernel.evaluate(set1)
        fastEval = object_sparse_py_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_sparse_py_kernel", functions, indirect=['object_sparse_py_kernel'])
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

    ############ 4 tests for clustered mode with internel clustering #######################

    @pytest.mark.parametrize("object_clustered_mode_birch", functions, indirect=['object_clustered_mode_birch'])
    def test_sparse_py_eval_groundset(self, object_clustered_mode_birch):
        groundSet = object_clustered_mode_birch.getEffectiveGroundSet()
        eval = object_clustered_mode_birch.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_clustered_mode_birch", functions, indirect=['object_clustered_mode_birch'])
    def test_sparse_py_eval_evalfast(self, data, object_clustered_mode_birch):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_mode_birch.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_mode_birch.evaluate(subset)
        fastEval = object_clustered_mode_birch.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_mode_birch", functions, indirect=['object_clustered_mode_birch'])
    def test_sparse_py_set_memoization(self, data, object_clustered_mode_birch):
        _, _, set1, _ = data
        object_clustered_mode_birch.setMemoization(set1)
        simpleEval = object_clustered_mode_birch.evaluate(set1)
        fastEval = object_clustered_mode_birch.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_mode_birch", functions, indirect=['object_clustered_mode_birch'])
    def test_sparse_py_gain(self, data, object_clustered_mode_birch):
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
    
    ############ 4 tests for clustered mode with user provided clustering #######################

    @pytest.mark.parametrize("object_clustered_mode_user", functions, indirect=['object_clustered_mode_user'])
    def test_sparse_py_eval_groundset(self, object_clustered_mode_user):
        groundSet = object_clustered_mode_user.getEffectiveGroundSet()
        eval = object_clustered_mode_user.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_clustered_mode_user", functions, indirect=['object_clustered_mode_user'])
    def test_sparse_py_eval_evalfast(self, data, object_clustered_mode_user):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_mode_user.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_mode_user.evaluate(subset)
        fastEval = object_clustered_mode_user.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_mode_user", functions, indirect=['object_clustered_mode_user'])
    def test_sparse_py_set_memoization(self, data, object_clustered_mode_user):
        _, _, set1, _ = data
        object_clustered_mode_user.setMemoization(set1)
        simpleEval = object_clustered_mode_user.evaluate(set1)
        fastEval = object_clustered_mode_user.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_mode_user", functions, indirect=['object_clustered_mode_user'])
    def test_sparse_py_gain(self, data, object_clustered_mode_user):
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

    ############ 4 tests for clustered function with internel clustering and multiple small kernels #######################

    @pytest.mark.parametrize("object_clustered_birch_multi", functions, indirect=['object_clustered_birch_multi'])
    def test_sparse_py_eval_groundset(self, object_clustered_birch_multi):
        groundSet = object_clustered_birch_multi.getEffectiveGroundSet()
        eval = object_clustered_birch_multi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_clustered_birch_multi", functions, indirect=['object_clustered_birch_multi'])
    def test_sparse_py_eval_evalfast(self, data, object_clustered_birch_multi):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_birch_multi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_birch_multi.evaluate(subset)
        fastEval = object_clustered_birch_multi.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_birch_multi", functions, indirect=['object_clustered_birch_multi'])
    def test_sparse_py_set_memoization(self, data, object_clustered_birch_multi):
        _, _, set1, _ = data
        object_clustered_birch_multi.setMemoization(set1)
        simpleEval = object_clustered_birch_multi.evaluate(set1)
        fastEval = object_clustered_birch_multi.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_birch_multi", functions, indirect=['object_clustered_birch_multi'])
    def test_sparse_py_gain(self, data, object_clustered_birch_multi):
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
    
    ############ 4 tests for clustered function with user provided clustering and multiple small kernels #######################

    @pytest.mark.parametrize("object_clustered_user_multi", functions, indirect=['object_clustered_user_multi'])
    def test_sparse_py_eval_groundset(self, object_clustered_user_multi):
        groundSet = object_clustered_user_multi.getEffectiveGroundSet()
        eval = object_clustered_user_multi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_clustered_user_multi", functions, indirect=['object_clustered_user_multi'])
    def test_sparse_py_eval_evalfast(self, data, object_clustered_user_multi):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_user_multi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_user_multi.evaluate(subset)
        fastEval = object_clustered_user_multi.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_user_multi", functions, indirect=['object_clustered_user_multi'])
    def test_sparse_py_set_memoization(self, data, object_clustered_user_multi):
        _, _, set1, _ = data
        object_clustered_user_multi.setMemoization(set1)
        simpleEval = object_clustered_user_multi.evaluate(set1)
        fastEval = object_clustered_user_multi.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_user_multi", functions, indirect=['object_clustered_user_multi'])
    def test_sparse_py_gain(self, data, object_clustered_user_multi):
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

    ############ 4 tests for clustered function with internel clustering and single kernel #######################

    @pytest.mark.parametrize("object_clustered_birch_single", functions, indirect=['object_clustered_birch_single'])
    def test_sparse_py_eval_groundset(self, object_clustered_birch_single):
        groundSet = object_clustered_birch_single.getEffectiveGroundSet()
        eval = object_clustered_birch_single.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_clustered_birch_single", functions, indirect=['object_clustered_birch_single'])
    def test_sparse_py_eval_evalfast(self, data, object_clustered_birch_single):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_birch_single.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_birch_single.evaluate(subset)
        fastEval = object_clustered_birch_single.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_birch_single", functions, indirect=['object_clustered_birch_single'])
    def test_sparse_py_set_memoization(self, data, object_clustered_birch_single):
        _, _, set1, _ = data
        object_clustered_birch_single.setMemoization(set1)
        simpleEval = object_clustered_birch_single.evaluate(set1)
        fastEval = object_clustered_birch_single.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_birch_single", functions, indirect=['object_clustered_birch_single'])
    def test_sparse_py_gain(self, data, object_clustered_birch_single):
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
    
    ############ 4 tests for clustered function with user provided clustering and single kernel #######################

    @pytest.mark.parametrize("object_clustered_user_single", functions, indirect=['object_clustered_user_single'])
    def test_sparse_py_eval_groundset(self, object_clustered_user_single):
        groundSet = object_clustered_user_single.getEffectiveGroundSet()
        eval = object_clustered_user_single.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.parametrize("object_clustered_user_single", functions, indirect=['object_clustered_user_single'])
    def test_sparse_py_eval_evalfast(self, data, object_clustered_user_single):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_clustered_user_single.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_clustered_user_single.evaluate(subset)
        fastEval = object_clustered_user_single.evaluateWithMemoization(subset)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.parametrize("object_clustered_user_single", functions, indirect=['object_clustered_user_single'])
    def test_sparse_py_set_memoization(self, data, object_clustered_user_single):
        _, _, set1, _ = data
        object_clustered_user_single.setMemoization(set1)
        simpleEval = object_clustered_user_single.evaluate(set1)
        fastEval = object_clustered_user_single.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.parametrize("object_clustered_user_single", functions, indirect=['object_clustered_user_single'])
    def test_sparse_py_gain(self, data, object_clustered_user_single):
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


    






