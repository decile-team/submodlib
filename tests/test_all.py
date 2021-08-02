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
from submodlib import FacilityLocationMutualInformationFunction
from submodlib import FacilityLocationVariantMutualInformationFunction
from submodlib import ConcaveOverModularFunction
from submodlib import GraphCutMutualInformationFunction
from submodlib import LogDeterminantMutualInformationFunction
from submodlib import GraphCutConditionalGainFunction
from submodlib import FacilityLocationConditionalGainFunction
from submodlib import LogDeterminantConditionalGainFunction
from submodlib import ProbabilisticSetCoverConditionalGainFunction
from submodlib import ProbabilisticSetCoverMutualInformationFunction
from submodlib import SetCoverMutualInformationFunction
from submodlib import SetCoverConditionalGainFunction
from submodlib import FacilityLocationConditionalMutualInformationFunction
from submodlib import LogDeterminantConditionalMutualInformationFunction
from submodlib import SetCoverConditionalMutualInformationFunction
from submodlib import ProbabilisticSetCoverConditionalMutualInformationFunction
import submodlib.helper as helper
from submodlib.helper import create_kernel
from submodlib_cpp import FeatureBased
from submodlib_cpp import ConcaveOverModular

#TODO: add test cases for rectangular kernels

allKernelFunctions = ["FacilityLocation", "DisparitySum", "GraphCut", "DisparityMin", "LogDeterminant"]
#allKernelFunctions = ["FacilityLocation", "DisparitySum", "GraphCut", "DisparityMin"]
#allKernelFunctions = ["LogDeterminant"]

allKernelMIFunctions = ["FacilityLocationMutualInformation", "FacilityLocationVariantMutualInformation", "ConcaveOverModular", "GraphCutMutualInformation", "GraphCutConditionalGain", "LogDeterminantMutualInformation", "FacilityLocationConditionalGain", "LogDeterminantConditionalGain"]
#allKernelMIFunctions = ["FacilityLocationMutualInformation", "FacilityLocationVariantMutualInformation", "ConcaveOverModular", "GraphCutMutualInformation", "GraphCutConditionalGain", "FacilityLocationConditionalGain"]
#allKernelMIFunctions = ["LogDeterminantMutualInformation", "LogDeterminantConditionalGain"]

allKernelCMIFunctions = ["FacilityLocationConditionalMutualInformation", "LogDeterminantConditionalMutualInformation"]
#allKernelCMIFunctions = ["FacilityLocationConditionalMutualInformation"]
#allKernelCMIFunctions = ["LogDeterminantConditionalMutualInformation"]

clusteredModeFunctions = ["FacilityLocation"]

optimizerTests = ["FacilityLocation", "GraphCut", "LogDeterminant"]
#optimizerTests = ["FacilityLocation", "GraphCut"]
#optimizerTests = ["LogDeterminant"]

optimizerMITests = ["FacilityLocationMutualInformation", "FacilityLocationVariantMutualInformation", "ConcaveOverModular", "GraphCutMutualInformation", "GraphCutConditionalGain", "LogDeterminantMutualInformation", "FacilityLocationConditionalGain", "LogDeterminantConditionalGain"]
#optimizerMITests = ["FacilityLocationMutualInformation", "FacilityLocationVariantMutualInformation", "ConcaveOverModular", "GraphCutMutualInformation", "GraphCutConditionalGain", "FacilityLocationConditionalGain"]
#optimizerMITests = ["LogDeterminantMutualInformation", "LogDeterminantConditionalGain"]

optimizerCMITests = ["FacilityLocationConditionalMutualInformation", "LogDeterminantConditionalMutualInformation"]
#optimizerCMITests = ["FacilityLocationConditionalMutualInformation"]
#optimizerCMITests = ["LogDeterminantConditionalMutualInformation"]

probSCMIFunctions = ["ProbabilisticSetCoverConditionalGain", "ProbabilisticSetCoverMutualInformation"]

SCMIFunctions = ["SetCoverMutualInformation", "SetCoverConditionalGain"]


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
# mi_regular - regular tests for functions listed in allKernelMIFunctions list
# mi_opt_regular - regular optimizer tests for functions listed in optimizerMITests list
# psc_mi_opt - for optimizer tests of PSC MI and CG functions
# psc_mi_regular - for regular tests of PSC MI and CG functions
# sc_mi_opt - for optimizer tests of SC MI and CG functions
# sc_mi_regular - for regular tests of SC MI and CG functions
# cmi_regular - for regular tests for CMI functions
# cmi_opt_regular - for optimizer tests for CMI functions
# sc_cmi_regular - for regular tests of SC CMI
# sc_cmi_opt - for optimizer tests of SC CMI
# psc_cmi_regular - for regular tests of PSC CMI
# psc_cmi_opt - for optimizer tests of PSC CMI
# cpp_kernel_cpp - for checking CPP kernel creation in CPP
# pybind_test - to check different alternatives of passing numpy array to C++
# single - makr any specific tests to run using this marker

# num_internal_clusters = 20 #3
# num_sparse_neighbors = 100 #10 #4
# num_random = 15 #2
# num_clusters = 20 #3
# cluster_std_dev = 4 #1
# num_samples = 500 #8
# num_set = 20 #3
# num_features = 500
# num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
# budget = 20
# num_concepts = 50
# num_queries = 10
# magnificationEta = 1 #3 #1 #3
# privacyHardness = 1 #3 #1 #3
# num_privates=5
# queryDiversityEta = 1
# logDetLambdaVal = 1
# metric = "euclidean"
# metric_disp_sparse = "euclidean"
# minbound = -10
# maxbound = 10

num_internal_clusters = 3 #3
num_sparse_neighbors = 5 #10 #4
num_random = 2 #2
num_clusters = 3#3
cluster_std_dev = 4 #1
num_samples = 9
num_set = 3 #3
num_features = 2
num_sparse_neighbors_full = num_sparse_neighbors #fixed sparseKernel asymmetric issue and hence this works for DisparitySum also now
budget = 5
num_concepts = 3
num_queries = 2
magnificationEta = 2
privacyHardness = 2
num_privates = 1
queryDiversityEta = 2
logDetLambdaVal = 1
metric = "euclidean"
metric_disp_sparse = "euclidean"
minbound = 50
maxbound = 200


@pytest.fixture
def data():
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(minbound,maxbound), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    #random.seed(1)
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
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(minbound,maxbound), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    # random.seed(1)
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
def data_queries():
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(minbound,maxbound), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    
    pointsMinusQuery = list(map(tuple, points)) 

    queries = []
    query_features = []
    # random.seed(1)
    random_clusters = random.sample(range(num_clusters), num_queries)
    for c in range(num_queries): #select 10 query points
        crand = random_clusters[c]
        q_ind = cluster_ids.tolist().index(crand) #find the ind of first point that belongs to cluster crand
        queries.append(q_ind)
        query_features.append(tuple(points[q_ind]))
        pointsMinusQuery.remove(tuple(points[q_ind]))
    
    # get a subset with num_set data points
    # random.seed(1)
    set1 = set(random.sample(range(num_samples-num_queries), num_set))

    imageData = np.array(pointsMinusQuery)
    queryData = np.array(query_features)

    return (num_samples-num_queries, num_queries, imageData, queryData, set1)

@pytest.fixture
def data_queries_privates():
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(minbound,maxbound), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    
    pointsMinusQueryPrivate = list(map(tuple, points)) 

    queries = []
    query_features = []
    privates = []
    private_features = []
    # random.seed(1)
    random_clusters = random.sample(range(num_clusters), num_queries + num_privates)
    for c in range(num_queries + num_privates): 
        if c < num_queries:
            crand = random_clusters[c]
            q_ind = cluster_ids.tolist().index(crand) #find the ind of first point that belongs to cluster crand
            queries.append(q_ind)
            query_features.append(tuple(points[q_ind]))
            pointsMinusQueryPrivate.remove(tuple(points[q_ind]))
        else:
            crand = random_clusters[c]
            p_ind = cluster_ids.tolist().index(crand) #find the ind of first point that belongs to cluster crand
            privates.append(p_ind)
            private_features.append(tuple(points[p_ind]))
            pointsMinusQueryPrivate.remove(tuple(points[p_ind]))
        
    # get a subset with num_set data points
    # random.seed(1)
    set1 = set(random.sample(range(num_samples-num_queries), num_set))

    imageData = np.array(pointsMinusQueryPrivate)
    queryData = np.array(query_features)
    privateData = np.array(private_features)

    return (num_samples-num_queries-num_privates, num_queries, num_privates, imageData, queryData, privateData, set1)
    

@pytest.fixture
def data_features_log():
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(minbound,maxbound), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    features = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    # random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    obj = FeatureBasedFunction(n=num_samples, features=features, numFeatures=num_features, sparse=False)

    return (obj, set1)

@pytest.fixture
def data_features_sqrt():
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(minbound,maxbound), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    features = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    # random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    obj = FeatureBasedFunction(n=num_samples, features=features, numFeatures=num_features, mode=FeatureBased.squareRoot, sparse=False)

    return (obj, set1)

@pytest.fixture
def data_features_inverse():
    random.seed(1)
    np.random.seed(1)
    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, center_box=(minbound,maxbound), cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    features = list(map(tuple, points))

    # get num_set data points belonging to cluster#1
    # random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    set1 = set(subset1[:-1])

    obj = FeatureBasedFunction(n=num_samples, features=features, numFeatures=num_features, mode=FeatureBased.inverse, sparse=False)

    return (obj, set1)


@pytest.fixture
def data_concepts():
    random.seed(1)
    np.random.seed(1)
    cover_set = []
    # np.random.seed(1)
    # random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        cover_set.append(set(random.sample(list(range(num_concepts)), random.randint(0,num_concepts))))
    obj = SetCoverFunction(n=num_samples, cover_set=cover_set, num_concepts=num_concepts, concept_weights=concept_weights)
    # random.seed(1)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def data_prob_concepts():
    random.seed(1)
    np.random.seed(1)
    probs = []
    # np.random.seed(1)
    # random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        probs.append(np.random.rand(num_concepts).tolist())
    obj = ProbabilisticSetCoverFunction(n=num_samples, probs=probs, num_concepts=num_concepts, concept_weights=concept_weights)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def data_mi_prob_concepts(request):
    random.seed(1)
    np.random.seed(1)
    probs = []
    # np.random.seed(1)
    # random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        probs.append(np.random.rand(num_concepts).tolist())
    privates = set(random.sample(range(num_concepts),num_queries))
    if request.param == "ProbabilisticSetCoverConditionalGain":
        obj = ProbabilisticSetCoverConditionalGainFunction(n=num_samples, probs=probs, num_concepts=num_concepts, concept_weights=concept_weights, private_concepts=privates)
    elif request.param == "ProbabilisticSetCoverMutualInformation":
        obj = ProbabilisticSetCoverMutualInformationFunction(n=num_samples, probs=probs, num_concepts=num_concepts, concept_weights=concept_weights, query_concepts=privates)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def data_mi_concepts(request):
    random.seed(1)
    np.random.seed(1)
    cover_set = []
    # np.random.seed(1)
    # random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        cover_set.append(set(random.sample(list(range(num_concepts)), random.randint(0,num_concepts))))
    queries = set(random.sample(range(num_concepts),num_queries))
    if request.param == "SetCoverMutualInformation":
        obj = SetCoverMutualInformationFunction(n=num_samples, cover_set=cover_set, num_concepts=num_concepts, concept_weights=concept_weights, query_concepts = queries)
    elif request.param == "SetCoverConditionalGain":
        obj = SetCoverConditionalGainFunction(n=num_samples, cover_set=cover_set, num_concepts=num_concepts, concept_weights=concept_weights, private_concepts = queries)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def data_cmi_concepts():
    random.seed(1)
    np.random.seed(1)
    cover_set = []
    # np.random.seed(1)
    # random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        cover_set.append(set(random.sample(list(range(num_concepts)), random.randint(0,num_concepts))))
    queries = set(random.sample(range(num_concepts),num_queries))
    set_concepts = set(range(num_concepts))
    set_concepts_minus_queries = set_concepts - queries
    privates = set(random.sample(set_concepts_minus_queries,num_privates))
    obj = SetCoverConditionalMutualInformationFunction(n=num_samples, cover_set=cover_set, num_concepts=num_concepts, concept_weights=concept_weights, query_concepts = queries, private_concepts=privates)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def data_cmi_prob_concepts():
    random.seed(1)
    np.random.seed(1)
    probs = []
    # np.random.seed(1)
    # random.seed(1)
    concept_weights = np.random.rand(num_concepts).tolist()
    for i in range(num_samples):
        probs.append(np.random.rand(num_concepts).tolist())
    queries = set(random.sample(range(num_concepts),num_queries))
    set_concepts = set(range(num_concepts))
    set_concepts_minus_queries = set_concepts - queries
    privates = set(random.sample(set_concepts_minus_queries,num_privates))
    obj = ProbabilisticSetCoverConditionalMutualInformationFunction(n=num_samples, probs=probs, num_concepts=num_concepts, concept_weights=concept_weights, query_concepts=queries, private_concepts=privates)
    subset1 = random.sample(list(range(num_samples)), num_set)
    set1 = set(subset1[:-1])
    return (obj, set1)

@pytest.fixture
def object_dense_cpp_kernel(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
    elif request.param == "DisparitySum":
        obj = DisparitySumFunction(n=num_samples, mode="dense", data=dataArray, metric=metric_disp_sparse)
    elif request.param == "DisparityMin":
        obj = DisparityMinFunction(n=num_samples, mode="dense", data=dataArray, metric=metric_disp_sparse)
    elif request.param == "GraphCut":
        obj = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, data=dataArray, metric=metric)
    elif request.param == "LogDeterminant":
        obj = LogDeterminantFunction(n=num_samples, mode="dense", lambdaVal=logDetLambdaVal, data=dataArray, metric=metric)
    else:
        return None
    return obj

@pytest.fixture
def object_dense_cpp_kernel_cpp(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric=metric, create_dense_cpp_kernel_in_python=False)
    else:
        return None
    return obj

@pytest.fixture
def object_mi_dense_cpp_kernel(request, data_queries):
    num_data, num_q, imageData, queryData, _ = data_queries
    if request.param == "FacilityLocationMutualInformation":
        obj = FacilityLocationMutualInformationFunction(n=num_data, num_queries=num_q, data=imageData, queryData=queryData, metric=metric, magnificationEta=magnificationEta)
    elif request.param == "FacilityLocationVariantMutualInformation":
        obj = FacilityLocationVariantMutualInformationFunction(n=num_data, num_queries=num_q, data=imageData, queryData=queryData, metric=metric, queryDiversityEta=queryDiversityEta)
    elif request.param == "ConcaveOverModular":
        obj = ConcaveOverModularFunction(n=num_data, num_queries=num_q, data=imageData, queryData=queryData, metric=metric, queryDiversityEta=queryDiversityEta, mode=ConcaveOverModular.logarithmic)
    elif request.param == "GraphCutMutualInformation":
        obj = GraphCutMutualInformationFunction(n=num_data, num_queries=num_q, data=imageData, queryData=queryData, metric=metric)
    elif request.param == "GraphCutConditionalGain":
        obj = GraphCutConditionalGainFunction(n=num_data, num_privates=num_q, lambdaVal=1, data=imageData, privateData=queryData, metric=metric, privacyHardness=privacyHardness)
    elif request.param == "FacilityLocationConditionalGain":
        obj = FacilityLocationConditionalGainFunction(n=num_data, num_privates=num_q, data=imageData, privateData=queryData, metric=metric, privacyHardness=privacyHardness)
    elif request.param == "LogDeterminantMutualInformation":
        obj = LogDeterminantMutualInformationFunction(n=num_data, num_queries=num_q, data=imageData, queryData=queryData, metric=metric, lambdaVal=logDetLambdaVal, magnificationEta=magnificationEta)
    elif request.param == "LogDeterminantConditionalGain":
        obj = LogDeterminantConditionalGainFunction(n=num_data, num_privates=num_q, data=imageData, privateData=queryData, metric=metric, lambdaVal=logDetLambdaVal, privacyHardness=privacyHardness)
    else:
        return None
    return obj

@pytest.fixture
def object_cmi_dense_cpp_kernel(request, data_queries_privates):
    num_data, num_q, num_p, imageData, queryData, privateData, _ = data_queries_privates
    if request.param == "FacilityLocationConditionalMutualInformation":
        obj = FacilityLocationConditionalMutualInformationFunction(n=num_data, num_queries=num_q, num_privates=num_p, data=imageData, queryData=queryData, privateData=privateData, metric=metric, magnificationEta=magnificationEta, privacyHardness=privacyHardness)
    elif request.param == "LogDeterminantConditionalMutualInformation":
        obj = LogDeterminantConditionalMutualInformationFunction(n=num_data, num_queries=num_q, num_privates=num_p, data=imageData, queryData=queryData, privateData=privateData, metric=metric, lambdaVal=logDetLambdaVal, magnificationEta=magnificationEta, privacyHardness=privacyHardness)
    else:
        return None
    return obj

@pytest.fixture
def object_mi_dense_py_kernel(request, data_queries):
    num_data, num_q, imageData, queryData, _ = data_queries
    imageKernel = create_kernel(imageData, mode="dense", metric=metric)
    queryKernel = create_kernel(queryData, mode="dense", metric=metric, X_rep=imageData)
    queryQueryKernel = create_kernel(queryData, mode="dense", metric=metric)
    if request.param == "FacilityLocationMutualInformation":
        obj = FacilityLocationMutualInformationFunction(n=num_data, num_queries=num_q, data_sijs=imageKernel, query_sijs=queryKernel, magnificationEta=magnificationEta)
    elif request.param == "FacilityLocationVariantMutualInformation":
        obj = FacilityLocationVariantMutualInformationFunction(n=num_data, num_queries=num_q, query_sijs=queryKernel, queryDiversityEta=queryDiversityEta)
    elif request.param == "ConcaveOverModular":
        obj = ConcaveOverModularFunction(n=num_data, num_queries=num_q, query_sijs=queryKernel, queryDiversityEta=queryDiversityEta, mode=ConcaveOverModular.logarithmic)
    elif request.param == "GraphCutMutualInformation":
        obj = GraphCutMutualInformationFunction(n=num_data, num_queries=num_q, query_sijs=queryKernel)
    elif request.param == "GraphCutConditionalGain":
        obj = GraphCutConditionalGainFunction(n=num_data, num_privates=num_q, lambdaVal=1, data_sijs=imageKernel, private_sijs=queryKernel, privacyHardness=privacyHardness)
    elif request.param == "FacilityLocationConditionalGain":
        obj = FacilityLocationConditionalGainFunction(n=num_data, num_privates=num_q, data_sijs=imageKernel, private_sijs=queryKernel, privacyHardness=privacyHardness)
    elif request.param == "LogDeterminantMutualInformation":
        obj = LogDeterminantMutualInformationFunction(n=num_data, num_queries=num_q, data_sijs=imageKernel, query_sijs=queryKernel, query_query_sijs=queryQueryKernel, lambdaVal=logDetLambdaVal, magnificationEta=magnificationEta)
    elif request.param == "LogDeterminantConditionalGain":
        obj = LogDeterminantConditionalGainFunction(n=num_data, num_privates=num_q, data_sijs=imageKernel, private_sijs=queryKernel, private_private_sijs=queryQueryKernel, lambdaVal=logDetLambdaVal, privacyHardness=privacyHardness)
    else:
        return None
    return obj

@pytest.fixture
def object_cmi_dense_py_kernel(request, data_queries_privates):
    num_data, num_q, num_p, imageData, queryData, privateData, _ = data_queries_privates
    imageKernel = create_kernel(imageData, mode="dense", metric=metric)
    queryKernel = create_kernel(queryData, mode="dense", metric=metric, X_rep=imageData)
    privateKernel = create_kernel(privateData, mode="dense", metric=metric, X_rep=imageData)
    queryQueryKernel = create_kernel(queryData, mode="dense", metric=metric)
    privatePrivateKernel = create_kernel(privateData, mode="dense", metric=metric)
    queryPrivateKernel = create_kernel(privateData, mode="dense", metric=metric, X_rep=queryData)
    if request.param == "FacilityLocationConditionalMutualInformation":
        obj = FacilityLocationConditionalMutualInformationFunction(n=num_data, num_queries=num_q, num_privates=num_p, data_sijs=imageKernel, query_sijs=queryKernel, private_sijs=privateKernel,magnificationEta=magnificationEta, privacyHardness=privacyHardness)
    elif request.param == "LogDeterminantConditionalMutualInformation":
        obj = LogDeterminantConditionalMutualInformationFunction(n=num_data, num_queries=num_q, num_privates=num_p, data_sijs=imageKernel, query_sijs=queryKernel, query_query_sijs=queryQueryKernel, private_sijs=privateKernel, private_private_sijs=privatePrivateKernel, query_private_sijs=queryPrivateKernel, lambdaVal=logDetLambdaVal, magnificationEta=magnificationEta, privacyHardness=privacyHardness)
    else:
        return None
    return obj

@pytest.fixture
def object_dense_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    K_dense = create_kernel(dataArray, mode='dense', metric=metric)
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False)
    elif request.param == "DisparitySum":
        K_dense = create_kernel(dataArray, mode='dense', metric=metric_disp_sparse)
        obj = DisparitySumFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "DisparityMin":
        K_dense = create_kernel(dataArray, mode='dense', metric=metric_disp_sparse)
        obj = DisparityMinFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "GraphCut":
        obj = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, ggsijs=K_dense)
    elif request.param == "LogDeterminant":
        obj = LogDeterminantFunction(n=num_samples, mode="dense", lambdaVal=logDetLambdaVal, sijs=K_dense)
    else:
        return None
    return obj

@pytest.fixture
def objects_dense_cpp_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    K_dense = create_kernel(dataArray, mode='dense',metric=metric)
    if request.param == "FacilityLocation":
        obj1 = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric=metric)
        obj2 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False)
    elif request.param == "DisparitySum":
        obj1 = DisparitySumFunction(n=num_samples, mode="dense", data=dataArray, metric=metric_disp_sparse)
        K_dense = create_kernel(dataArray, mode='dense',metric=metric_disp_sparse)
        obj2 = DisparitySumFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "DisparityMin":
        obj1 = DisparityMinFunction(n=num_samples, mode="dense", data=dataArray, metric=metric_disp_sparse)
        K_dense = create_kernel(dataArray, mode='dense',metric=metric_disp_sparse)
        obj2 = DisparityMinFunction(n=num_samples, mode="dense", sijs = K_dense)
    elif request.param == "GraphCut":
        obj1 = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, data=dataArray, metric=metric)
        obj2 = GraphCutFunction(n=num_samples, mode="dense", lambdaVal=1, ggsijs=K_dense)
    elif request.param == "LogDeterminant":
        obj1 = LogDeterminantFunction(n=num_samples, mode="dense", data=dataArray, metric=metric, lambdaVal=logDetLambdaVal)
        obj2 = LogDeterminantFunction(n=num_samples, mode="dense", sijs = K_dense, lambdaVal=logDetLambdaVal)
    else:
        return None
    return obj1, obj2

@pytest.fixture
def objects_fl_pybind(data):
    num_samples, dataArray, _, _ = data
    K_dense = helper.create_kernel_dense_sklearn(dataArray, metric)
    obj1 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False, pybind_mode="list")
    # obj2 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False, pybind_mode="memoryview")
    obj2 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False, pybind_mode="numpyarray")
    obj3 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False, pybind_mode="array")
    obj4 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False, pybind_mode="array32")
    obj5 = FacilityLocationFunction(n=num_samples, mode="dense", sijs = K_dense, separate_rep=False, pybind_mode="array64")
    return obj1, obj2, obj3, obj4, obj5

@pytest.fixture
def object_sparse_cpp_kernel(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)
    elif request.param == "DisparitySum":
        obj = DisparitySumFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric_disp_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "DisparityMin":
        obj = DisparityMinFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric_disp_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "GraphCut":
        obj = GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors)
    elif request.param == "LogDeterminant":
        obj = LogDeterminantFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric, num_neighbors=num_sparse_neighbors_full, lambdaVal=logDetLambdaVal)
    else:
        return None
    return obj

@pytest.fixture
def object_sparse_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    K_sparse = create_kernel(dataArray, mode='sparse',metric=metric_disp_sparse, num_neigh=num_sparse_neighbors)
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "DisparitySum":
        obj = DisparitySumFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "DisparityMin":
        obj = DisparityMinFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "GraphCut":
        obj = GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, ggsijs=K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "LogDeterminant":
        obj = LogDeterminantFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full, lambdaVal=logDetLambdaVal)
    else:
        return None
    return obj

@pytest.fixture
def objects_sparse_cpp_py_kernel(request, data):
    num_samples, dataArray, _, _ = data
    K_sparse = create_kernel(dataArray, mode='sparse', metric=metric_disp_sparse, num_neigh=num_sparse_neighbors)
    if request.param == "FacilityLocation":
        obj1 = FacilityLocationFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric_disp_sparse, num_neighbors=num_sparse_neighbors)
        obj2 = FacilityLocationFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "DisparitySum":
        obj1 = DisparitySumFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric_disp_sparse, num_neighbors=num_sparse_neighbors_full)
        obj2 = DisparitySumFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "DisparityMin":
        obj1 = DisparityMinFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric_disp_sparse, num_neighbors=num_sparse_neighbors_full)
        obj2 = DisparityMinFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full)
    elif request.param == "GraphCut":
        obj1= GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, data=dataArray, metric=metric_disp_sparse, num_neighbors=num_sparse_neighbors)
        obj2 = GraphCutFunction(n=num_samples, mode="sparse", lambdaVal=1, ggsijs=K_sparse, num_neighbors=num_sparse_neighbors)
    elif request.param == "LogDeterminant":
        obj1 = LogDeterminantFunction(n=num_samples, mode="sparse", data=dataArray, metric=metric_disp_sparse, num_neighbors=num_sparse_neighbors_full, lambdaVal=logDetLambdaVal)
        obj2 = LogDeterminantFunction(n=num_samples, mode="sparse", sijs = K_sparse, num_neighbors=num_sparse_neighbors_full, lambdaVal=logDetLambdaVal)
    else:
        return None
    return obj1, obj2

@pytest.fixture
def object_clustered_mode_birch(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric=metric_disp_sparse, num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_mode_user(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    if request.param == "FacilityLocation":
        obj = FacilityLocationFunction(n=num_samples, mode="clustered", cluster_labels=cluster_ids.tolist(), data=dataArray, metric=metric_disp_sparse, num_clusters=num_internal_clusters)
    else:
        return None
    return obj

@pytest.fixture
def object_clustered_birch_multi(request, data):
    num_samples, dataArray, _, _ = data
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters)
    return obj

@pytest.fixture
def object_clustered_user_multi(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    obj = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    return obj

@pytest.fixture
def object_clustered_birch_single(request, data):
    num_samples, dataArray, _, _ = data
    obj = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters)
    return obj

@pytest.fixture
def object_clustered_user_single(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    obj = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    return obj

@pytest.fixture
def objects_single_multi_user(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    obj1 = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    obj2 = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    return obj1, obj2

@pytest.fixture
def objects_mode_clustered_user(request, data_with_clusters):
    num_samples, dataArray, _, _, cluster_ids = data_with_clusters
    if request.param == "FacilityLocation":
        obj1 = FacilityLocationFunction(n=num_samples, mode="clustered", cluster_labels=cluster_ids.tolist(), data=dataArray, metric=metric_disp_sparse, num_clusters=num_internal_clusters)
        obj2 = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters, cluster_lab=cluster_ids.tolist())
    else:
        return None
    return obj1, obj2

@pytest.fixture
def objects_single_multi_birch(request, data):
    num_samples, dataArray, _, _ = data
    obj1 = ClusteredFunction(n=num_samples, mode="multi", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters)
    obj2 = ClusteredFunction(n=num_samples, mode="single", f_name=request.param, metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters)
    return obj1, obj2

@pytest.fixture
def objects_mode_clustered_birch(request, data):
    num_samples, dataArray, _, _ = data
    if request.param == "FacilityLocation":
        obj1 = FacilityLocationFunction(n=num_samples, mode="clustered", data=dataArray, metric=metric_disp_sparse, num_clusters=num_internal_clusters)
        obj2 = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', metric=metric_disp_sparse, data=dataArray, num_clusters=num_internal_clusters)
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
        elem = random.sample(set1, 1)[0]
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
    
    ############ 6 tests for dense cpp kernel made in CPP #######################
    @pytest.mark.cpp_kernel_cpp
    @pytest.mark.parametrize("object_dense_cpp_kernel_cpp", clusteredModeFunctions, indirect=['object_dense_cpp_kernel_cpp'])
    def test_dense_cpp_cpp_eval_groundset(self, object_dense_cpp_kernel_cpp):
        groundSet = object_dense_cpp_kernel_cpp.getEffectiveGroundSet()
        eval = object_dense_cpp_kernel_cpp.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.cpp_kernel_cpp
    @pytest.mark.parametrize("object_dense_cpp_kernel_cpp", clusteredModeFunctions, indirect=['object_dense_cpp_kernel_cpp'])
    def test_dense_cpp_cpp_eval_emptyset(self, object_dense_cpp_kernel_cpp):
        eval = object_dense_cpp_kernel_cpp.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.cpp_kernel_cpp
    @pytest.mark.parametrize("object_dense_cpp_kernel_cpp", clusteredModeFunctions, indirect=['object_dense_cpp_kernel_cpp'])
    def test_dense_cpp_cpp_gain_on_empty(self, data, object_dense_cpp_kernel_cpp):
        _, _, set1, _ = data
        elem = random.sample(set1, 1)[0]
        testSet = set()
        evalEmpty = object_dense_cpp_kernel_cpp.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_dense_cpp_kernel_cpp.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_dense_cpp_kernel_cpp.marginalGain(set(), elem)
        assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.cpp_kernel_cpp
    @pytest.mark.parametrize("object_dense_cpp_kernel_cpp", clusteredModeFunctions, indirect=['object_dense_cpp_kernel_cpp'])
    def test_dense_cpp_cpp_eval_evalfast(self, data, object_dense_cpp_kernel_cpp):
        _, _, set1, _ = data
        subset = set()
        for elem in set1:
            object_dense_cpp_kernel_cpp.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_dense_cpp_kernel_cpp.evaluate(subset)
        fastEval = object_dense_cpp_kernel_cpp.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.cpp_kernel_cpp
    @pytest.mark.parametrize("object_dense_cpp_kernel_cpp", clusteredModeFunctions, indirect=['object_dense_cpp_kernel_cpp'])
    def test_dense_cpp_cpp_set_memoization(self, data, object_dense_cpp_kernel_cpp):
        _, _, set1, _ = data
        object_dense_cpp_kernel_cpp.setMemoization(set1)
        simpleEval = object_dense_cpp_kernel_cpp.evaluate(set1)
        fastEval = object_dense_cpp_kernel_cpp.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.cpp_kernel_cpp
    @pytest.mark.parametrize("object_dense_cpp_kernel_cpp", clusteredModeFunctions, indirect=['object_dense_cpp_kernel_cpp'])
    def test_dense_cpp_cpp_gain(self, data, object_dense_cpp_kernel_cpp):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_dense_cpp_kernel_cpp.setMemoization(subset)
        firstEval = object_dense_cpp_kernel_cpp.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_dense_cpp_kernel_cpp.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_dense_cpp_kernel_cpp.marginalGain(subset, elem)
        fastGain = object_dense_cpp_kernel_cpp.marginalGainWithMemoization(subset, elem)
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
        elem = random.sample(set1, 1)[0]
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
        elem = random.sample(set1, 1)[0]
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
        elem = random.sample(set1, 1)[0]
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
        elem = random.sample(set1, 1)[0]
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
        elem = random.sample(set1, 1)[0]
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
        elem = random.sample(set1, 1)[0]
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
        elem = random.sample(set1, 1)[0]
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

    ############ 7 tests for clustered function with internel clustering and single kernel #######################

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
        elem = random.sample(set1, 1)[0]
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
        elem = random.sample(set1, 1)[0]
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

    ######### 4 Tests to check the consistency between different pybind alternatives

    @pytest.mark.pybind_test
    @pytest.mark.parametrize("objects_fl_pybind", clusteredModeFunctions, indirect=['objects_fl_pybind'])
    def test_pybind_eval(self, data, objects_fl_pybind):
        _, _, set1, _ = data
        eval1 = objects_fl_pybind[0].evaluate(set1)
        eval2 = objects_fl_pybind[1].evaluate(set1)
        eval3 = objects_fl_pybind[2].evaluate(set1)
        eval4 = objects_fl_pybind[3].evaluate(set1)
        eval5 = objects_fl_pybind[4].evaluate(set1)
        assert math.isclose(eval1, eval2, rel_tol=1e-05) and math.isclose(eval2, eval3, rel_tol=1e-05) and math.isclose(eval3, eval4, rel_tol=1e-05) and math.isclose(eval4, eval5, rel_tol=1e-05), "Mismatch between evaluate() of different pybind alternatives"
    
    @pytest.mark.pybind_test
    @pytest.mark.parametrize("objects_fl_pybind", clusteredModeFunctions, indirect=['objects_fl_pybind'])
    def test_pybind_gain(self, data, objects_fl_pybind):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        gain1 = objects_fl_pybind[0].marginalGain(subset, elem)
        gain2 = objects_fl_pybind[1].marginalGain(subset, elem)
        gain3 = objects_fl_pybind[2].marginalGain(subset, elem)
        gain4 = objects_fl_pybind[3].marginalGain(subset, elem)
        gain5 = objects_fl_pybind[4].marginalGain(subset, elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05) and math.isclose(gain2, gain3, rel_tol=1e-05) and math.isclose(gain3, gain4, rel_tol=1e-05) and math.isclose(gain4, gain5, rel_tol=1e-05), "Mismatch between marginalGain() of different pybind alternatives"
    
    @pytest.mark.pybind_test
    @pytest.mark.parametrize("objects_fl_pybind", clusteredModeFunctions, indirect=['objects_fl_pybind'])
    def test_pybind_evalFast(self, data, objects_fl_pybind):
        _, _, set1, _ = data
        objects_fl_pybind[0].setMemoization(set1)
        objects_fl_pybind[1].setMemoization(set1)
        objects_fl_pybind[2].setMemoization(set1)
        objects_fl_pybind[3].setMemoization(set1)
        objects_fl_pybind[4].setMemoization(set1)
        evalFast1 = objects_fl_pybind[0].evaluateWithMemoization(set1)
        evalFast2 = objects_fl_pybind[1].evaluateWithMemoization(set1)
        evalFast3 = objects_fl_pybind[2].evaluateWithMemoization(set1)
        evalFast4 = objects_fl_pybind[3].evaluateWithMemoization(set1)
        evalFast5= objects_fl_pybind[4].evaluateWithMemoization(set1)
        assert math.isclose(evalFast1, evalFast2, rel_tol=1e-05) and math.isclose(evalFast2, evalFast3, rel_tol=1e-05) and math.isclose(evalFast3, evalFast4, rel_tol=1e-05) and math.isclose(evalFast4, evalFast5, rel_tol=1e-05), "Mismatch between evaluateWithMemoization() of different pybind alternatives"
    
    @pytest.mark.pybind_test
    @pytest.mark.parametrize("objects_fl_pybind", clusteredModeFunctions, indirect=['objects_fl_pybind'])
    def test_pybind_gainFast(self, data, objects_fl_pybind):
        _, _, set1, _ = data
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        objects_fl_pybind[0].setMemoization(subset)
        objects_fl_pybind[1].setMemoization(subset)
        objects_fl_pybind[2].setMemoization(subset)
        objects_fl_pybind[3].setMemoization(subset)
        objects_fl_pybind[4].setMemoization(subset)
        gainFast1 = objects_fl_pybind[0].marginalGainWithMemoization(subset, elem)
        gainFast2 = objects_fl_pybind[1].marginalGainWithMemoization(subset, elem)
        gainFast3 = objects_fl_pybind[2].marginalGainWithMemoization(subset, elem)
        gainFast4 = objects_fl_pybind[3].marginalGainWithMemoization(subset, elem)
        gainFast5 = objects_fl_pybind[4].marginalGainWithMemoization(subset, elem)
        assert math.isclose(gainFast1, gainFast2, rel_tol=1e-05) and math.isclose(gainFast2, gainFast3, rel_tol=1e-05) and math.isclose(gainFast3, gainFast4, rel_tol=1e-05) and math.isclose(gainFast4, gainFast5, rel_tol=1e-05) , "Mismatch between marginalGainWithMemoization() of different pybind alternatives"

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
    @pytest.mark.parametrize("object_dense_py_kernel", optimizerTests, indirect=['object_dense_py_kernel'])
    def test_naive_lazy(self, object_dense_py_kernel):
        greedyListNaive = object_dense_py_kernel.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_dense_py_kernel.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        # assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.opt_regular
    @pytest.mark.parametrize("object_dense_py_kernel", optimizerTests, indirect=['object_dense_py_kernel'])
    def test_stochastic_lazierThanLazy(self, object_dense_py_kernel):
        greedyListStochastic = object_dense_py_kernel.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_dense_py_kernel.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ####### Sanity check of cost sensitive versions #############
    @pytest.mark.opt_regular
    @pytest.mark.parametrize("object_dense_py_kernel", optimizerTests, indirect=['object_dense_py_kernel'])
    def test_naive_naive_cost(self, object_dense_py_kernel):
        costs = [1]*num_samples
        greedyListNaive1 = object_dense_py_kernel.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListNaive2 = object_dense_py_kernel.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, costs = costs, costSensitiveGreedy=True)
        naiveGains1 = [x[1] for x in greedyListNaive1]
        naiveGains2 = [x[1] for x in greedyListNaive2]
        # assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
        assert naiveGains1 == pytest.approx(naiveGains2), "Mismatch between cost sensitive and cost agnostic versions of naive greedy"
    
    @pytest.mark.opt_regular
    @pytest.mark.parametrize("object_dense_py_kernel", optimizerTests, indirect=['object_dense_py_kernel'])
    def test_naive_lazy_cost(self, object_dense_py_kernel):
        costs = random.choices(range(1,6), k=num_samples)
        greedyListNaive = object_dense_py_kernel.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, costs=costs, costSensitiveGreedy=True)
        greedyListLazy = object_dense_py_kernel.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False, costs=costs, costSensitiveGreedy=True)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        # assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy cost sensitive and lazyGreedy cost sensitive"
    

    ######## Optimizers test for FeatureBased Logarithmic #####################
    @pytest.mark.fb_opt
    def test_fb_log_optimizer_naive_lazy(self, data_features_log):
        object_fb, _ = data_features_log
        greedyListNaive = object_fb.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_fb.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        # assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.fb_opt
    def test_fb_log_optimizer_stochastic_lazierThanLazy(self, data_features_log):
        object_fb, _ = data_features_log
        greedyListStochastic = object_fb.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_fb.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

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
        elem = random.sample(set1, 1)[0]
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
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        # assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.fb_opt
    def test_fb_sqrt_optimizer_stochastic_lazierThanLazy(self, data_features_sqrt):
        object_fb, _ = data_features_sqrt
        greedyListStochastic = object_fb.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_fb.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

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
        elem = random.sample(set1, 1)[0]
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
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        # assert naiveGains == lazyGains, "Mismatch between naiveGreedy and lazyGreedy"
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.fb_opt
    def test_fb_inverse_optimizer_stochastic_lazierThanLazy(self, data_features_inverse):
        object_fb, _ = data_features_inverse
        greedyListStochastic = object_fb.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_fb.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

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
        elem = random.sample(set1, 1)[0]
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
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.sc_opt
    def test_sc_optimizer_stochastic_lazierThanLazy(self, data_concepts):
        object_sc, _ = data_concepts
        greedyListStochastic = object_sc.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_sc.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

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
        elem = random.sample(set1, 1)[0]
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
    
    ######## Optimizers test for SetCoverMI #####################
    @pytest.mark.sc_mi_opt
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_optimizer_naive_lazy(self, data_mi_concepts):
        object_scmi, _ = data_mi_concepts
        greedyListNaive = object_scmi.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_scmi.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.sc_mi_opt
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_optimizer_stochastic_lazierThanLazy(self, data_mi_concepts):
        object_scmi, _ = data_mi_concepts
        greedyListStochastic = object_scmi.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_scmi.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for SetCover MI Function #######################
    @pytest.mark.sc_mi_regular
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_eval_groundset(self, data_mi_concepts):
        object_scmi, _ = data_mi_concepts
        groundSet = object_scmi.getEffectiveGroundSet()
        eval = object_scmi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.sc_mi_regular
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_eval_emptyset(self, data_mi_concepts):
        object_scmi, _ = data_mi_concepts
        eval = object_scmi.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.sc_mi_regular
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_gain_on_empty(self, data_mi_concepts):
        object_scmi, set1 = data_mi_concepts
        elem = random.sample(set1, 1)[0]
        testSet = set()
        evalEmpty = object_scmi.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_scmi.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_scmi.marginalGain(set(), elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch for gain on empty set"
        #assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.sc_mi_regular
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_eval_evalfast(self, data_mi_concepts):
        object_scmi, set1 = data_mi_concepts
        subset = set()
        for elem in set1:
            object_scmi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_scmi.evaluate(subset)
        fastEval = object_scmi.evaluateWithMemoization(subset)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.sc_mi_regular
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_set_memoization(self, data_mi_concepts):
        object_scmi, set1 = data_mi_concepts
        object_scmi.setMemoization(set1)
        simpleEval = object_scmi.evaluate(set1)
        fastEval = object_scmi.evaluateWithMemoization(set1)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.sc_mi_regular
    @pytest.mark.parametrize("data_mi_concepts", SCMIFunctions, indirect=['data_mi_concepts'])
    def test_sc_mi_gain(self, data_mi_concepts):
        object_scmi, set1 = data_mi_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_scmi.setMemoization(subset)
        firstEval = object_scmi.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_scmi.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_scmi.marginalGain(subset, elem)
        fastGain = object_scmi.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
        #assert naiveGain == simpleGain and simpleGain == fastGain, "Mismatch between naive, simple and fast margins"
    
    ######## Optimizers test for ProbabilisticSetCover #####################
    @pytest.mark.psc_opt
    def test_psc_optimizer_naive_lazy(self, data_prob_concepts):
        object_psc, _ = data_prob_concepts
        greedyListNaive = object_psc.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_psc.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.psc_opt
    def test_psc_optimizer_stochastic_lazierThanLazy(self, data_prob_concepts):
        object_psc, _ = data_prob_concepts
        greedyListStochastic = object_psc.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_psc.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

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
        elem = random.sample(set1, 1)[0]
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

    
    ######## Optimizers test for ProbabilisticSetCoverMI #####################
    @pytest.mark.psc_mi_opt
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_optimizer_naive_lazy(self, data_mi_prob_concepts):
        object_pscmi, _ = data_mi_prob_concepts
        greedyListNaive = object_pscmi.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_pscmi.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.psc_mi_opt
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_optimizer_stochastic_lazierThanLazy(self, data_mi_prob_concepts):
        object_pscmi, _ = data_mi_prob_concepts
        greedyListStochastic = object_pscmi.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_pscmi.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for ProbabilisticSetCover MI Function #######################
    @pytest.mark.psc_mi_regular
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_eval_groundset(self, data_mi_prob_concepts):
        object_pscmi, _ = data_mi_prob_concepts
        groundSet = object_pscmi.getEffectiveGroundSet()
        eval = object_pscmi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.psc_mi_regular
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_eval_emptyset(self, data_mi_prob_concepts):
        object_pscmi, _ = data_mi_prob_concepts
        eval = object_pscmi.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.psc_mi_regular
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_gain_on_empty(self, data_mi_prob_concepts):
        object_pscmi, set1 = data_mi_prob_concepts
        elem = random.sample(set1, 1)[0]
        testSet = set()
        evalEmpty = object_pscmi.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_pscmi.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_pscmi.marginalGain(set(), elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch for gain on empty set"
        #assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.psc_mi_regular
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_eval_evalfast(self, data_mi_prob_concepts):
        object_pscmi, set1 = data_mi_prob_concepts
        subset = set()
        for elem in set1:
            object_pscmi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_pscmi.evaluate(subset)
        fastEval = object_pscmi.evaluateWithMemoization(subset)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.psc_mi_regular
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_set_memoization(self, data_mi_prob_concepts):
        object_pscmi, set1 = data_mi_prob_concepts
        object_pscmi.setMemoization(set1)
        simpleEval = object_pscmi.evaluate(set1)
        fastEval = object_pscmi.evaluateWithMemoization(set1)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.psc_mi_regular
    @pytest.mark.parametrize("data_mi_prob_concepts", probSCMIFunctions, indirect=['data_mi_prob_concepts'])
    def test_psc_mi_gain(self, data_mi_prob_concepts):
        object_pscmi, set1 = data_mi_prob_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_pscmi.setMemoization(subset)
        firstEval = object_pscmi.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_pscmi.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_pscmi.marginalGain(subset, elem)
        fastGain = object_pscmi.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
        #assert naiveGain == simpleGain and simpleGain == fastGain, "Mismatch between naive, simple and fast margins"


    ############ 6 tests for MI dense cpp kernel #######################
    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_cpp_kernel", allKernelMIFunctions, indirect=['object_mi_dense_cpp_kernel'])
    def test_mi_dense_cpp_eval_groundset(self, object_mi_dense_cpp_kernel):
        groundSet = object_mi_dense_cpp_kernel.getEffectiveGroundSet()
        eval = object_mi_dense_cpp_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_cpp_kernel", allKernelMIFunctions, indirect=['object_mi_dense_cpp_kernel'])
    def test_mi_dense_cpp_eval_emptyset(self, object_mi_dense_cpp_kernel):
        eval = object_mi_dense_cpp_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_cpp_kernel", allKernelMIFunctions, indirect=['object_mi_dense_cpp_kernel'])
    def test_mi_dense_cpp_gain_on_empty(self, data_queries, object_mi_dense_cpp_kernel):
        _, _, _, _, set1 = data_queries
        elem = random.sample(set1, 1)[0]
        testSet = set()
        evalEmpty = object_mi_dense_cpp_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_mi_dense_cpp_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_mi_dense_cpp_kernel.marginalGain(set(), elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch for gain on empty set"

    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_cpp_kernel", allKernelMIFunctions, indirect=['object_mi_dense_cpp_kernel'])
    def test_mi_dense_cpp_eval_evalfast(self, data_queries, object_mi_dense_cpp_kernel):
        _, _, _, _, set1 = data_queries
        subset = set()
        for elem in set1:
            object_mi_dense_cpp_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_mi_dense_cpp_kernel.evaluate(subset)
        fastEval = object_mi_dense_cpp_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_cpp_kernel", allKernelMIFunctions, indirect=['object_mi_dense_cpp_kernel'])
    def test_mi_dense_cpp_set_memoization(self, data_queries, object_mi_dense_cpp_kernel):
        _, _, _, _, set1 = data_queries
        object_mi_dense_cpp_kernel.setMemoization(set1)
        simpleEval = object_mi_dense_cpp_kernel.evaluate(set1)
        fastEval = object_mi_dense_cpp_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_cpp_kernel", allKernelMIFunctions, indirect=['object_mi_dense_cpp_kernel'])
    def test_mi_dense_cpp_gain(self, data_queries, object_mi_dense_cpp_kernel):
        _, _, _, _, set1 = data_queries
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_mi_dense_cpp_kernel.setMemoization(subset)
        firstEval = object_mi_dense_cpp_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_mi_dense_cpp_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_mi_dense_cpp_kernel.marginalGain(subset, elem)
        fastGain = object_mi_dense_cpp_kernel.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"

    ############ 6 tests for dense python kernel #######################

    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", allKernelMIFunctions, indirect=['object_mi_dense_py_kernel'])
    def test_mi_dense_py_eval_groundset(self, object_mi_dense_py_kernel):
        groundSet = object_mi_dense_py_kernel.getEffectiveGroundSet()
        eval = object_mi_dense_py_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", allKernelMIFunctions, indirect=['object_mi_dense_py_kernel'])
    def test_mi_dense_py_eval_emptyset(self, object_mi_dense_py_kernel):
        eval = object_mi_dense_py_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", allKernelMIFunctions, indirect=['object_mi_dense_py_kernel'])
    def test_mi_dense_py_gain_on_empty(self, data_queries, object_mi_dense_py_kernel):
        _, _, _, _, set1 = data_queries
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_mi_dense_py_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_mi_dense_py_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_mi_dense_py_kernel.marginalGain(set(), elem)
        assert math.isclose(gain1,gain2,rel_tol=1e-05), "Mismatch for gain on empty set"
    
    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", allKernelMIFunctions, indirect=['object_mi_dense_py_kernel'])
    def test_mi_dense_py_eval_evalfast(self, data_queries, object_mi_dense_py_kernel):
        _, _, _, _, set1 = data_queries
        subset = set()
        for elem in set1:
            object_mi_dense_py_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_mi_dense_py_kernel.evaluate(subset)
        fastEval = object_mi_dense_py_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval,rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", allKernelMIFunctions, indirect=['object_mi_dense_py_kernel'])
    def test_mi_dense_py_set_memoization(self, data_queries, object_mi_dense_py_kernel):
        _, _, _, _, set1 = data_queries
        object_mi_dense_py_kernel.setMemoization(set1)
        simpleEval = object_mi_dense_py_kernel.evaluate(set1)
        fastEval = object_mi_dense_py_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.mi_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", allKernelMIFunctions, indirect=['object_mi_dense_py_kernel'])
    def test_mi_dense_py_gain(self, data_queries, object_mi_dense_py_kernel):
        _, _, _, _, set1 = data_queries
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_mi_dense_py_kernel.setMemoization(subset)
        firstEval = object_mi_dense_py_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_mi_dense_py_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_mi_dense_py_kernel.marginalGain(subset, elem)
        fastGain = object_mi_dense_py_kernel.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"

    ######## Tests for MI optimizers ################
    @pytest.mark.mi_opt_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", optimizerMITests, indirect=['object_mi_dense_py_kernel'])
    def test_mi_naive_lazy(self, object_mi_dense_py_kernel):
        greedyListNaive = object_mi_dense_py_kernel.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_mi_dense_py_kernel.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.mi_opt_regular
    @pytest.mark.parametrize("object_mi_dense_py_kernel", optimizerMITests, indirect=['object_mi_dense_py_kernel'])
    def test_mi_stochastic_lazierThanLazy(self, object_mi_dense_py_kernel):
        greedyListStochastic = object_mi_dense_py_kernel.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_mi_dense_py_kernel.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 tests for CMI dense cpp kernel #######################
    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_cpp_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_cpp_kernel'])
    def test_cmi_dense_cpp_eval_groundset(self, object_cmi_dense_cpp_kernel):
        groundSet = object_cmi_dense_cpp_kernel.getEffectiveGroundSet()
        eval = object_cmi_dense_cpp_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_cpp_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_cpp_kernel'])
    def test_cmi_dense_cpp_eval_emptyset(self, object_cmi_dense_cpp_kernel):
        eval = object_cmi_dense_cpp_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_cpp_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_cpp_kernel'])
    def test_cmi_dense_cpp_gain_on_empty(self, data_queries_privates, object_cmi_dense_cpp_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        elem = random.sample(set1, 1)[0]
        testSet = set()
        evalEmpty = object_cmi_dense_cpp_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_cmi_dense_cpp_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_cmi_dense_cpp_kernel.marginalGain(set(), elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch for gain on empty set"

    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_cpp_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_cpp_kernel'])
    def test_cmi_dense_cpp_eval_evalfast(self, data_queries_privates, object_cmi_dense_cpp_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        subset = set()
        for elem in set1:
            object_cmi_dense_cpp_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_cmi_dense_cpp_kernel.evaluate(subset)
        fastEval = object_cmi_dense_cpp_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_cpp_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_cpp_kernel'])
    def test_cmi_dense_cpp_set_memoization(self, data_queries_privates, object_cmi_dense_cpp_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        object_cmi_dense_cpp_kernel.setMemoization(set1)
        simpleEval = object_cmi_dense_cpp_kernel.evaluate(set1)
        fastEval = object_cmi_dense_cpp_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_cpp_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_cpp_kernel'])
    def test_cmi_dense_cpp_gain(self, data_queries_privates, object_cmi_dense_cpp_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_cmi_dense_cpp_kernel.setMemoization(subset)
        firstEval = object_cmi_dense_cpp_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_cmi_dense_cpp_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_cmi_dense_cpp_kernel.marginalGain(subset, elem)
        fastGain = object_cmi_dense_cpp_kernel.marginalGainWithMemoization(subset, elem)
        # assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
        assert naiveGain == pytest.approx(simpleGain) and simpleGain == pytest.approx(fastGain), "Mismatch between naive, simple and fast margins"

    ############ 6 tests for dense python kernel #######################

    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_dense_py_eval_groundset(self, object_cmi_dense_py_kernel):
        groundSet = object_cmi_dense_py_kernel.getEffectiveGroundSet()
        eval = object_cmi_dense_py_kernel.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"
    
    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_dense_py_eval_emptyset(self, object_cmi_dense_py_kernel):
        eval = object_cmi_dense_py_kernel.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_dense_py_gain_on_empty(self, data_queries_privates, object_cmi_dense_py_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        testSet = set()
        evalEmpty = object_cmi_dense_py_kernel.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_cmi_dense_py_kernel.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_cmi_dense_py_kernel.marginalGain(set(), elem)
        assert math.isclose(gain1,gain2,rel_tol=1e-05), "Mismatch for gain on empty set"
    
    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_dense_py_eval_evalfast(self, data_queries_privates, object_cmi_dense_py_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        subset = set()
        for elem in set1:
            object_cmi_dense_py_kernel.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_cmi_dense_py_kernel.evaluate(subset)
        fastEval = object_cmi_dense_py_kernel.evaluateWithMemoization(subset)
        assert math.isclose(simpleEval, fastEval,rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_dense_py_set_memoization(self, data_queries_privates, object_cmi_dense_py_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        object_cmi_dense_py_kernel.setMemoization(set1)
        simpleEval = object_cmi_dense_py_kernel.evaluate(set1)
        fastEval = object_cmi_dense_py_kernel.evaluateWithMemoization(set1)
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.cmi_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", allKernelCMIFunctions, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_dense_py_gain(self, data_queries_privates, object_cmi_dense_py_kernel):
        _, _, _, _, _, _, set1 = data_queries_privates
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_cmi_dense_py_kernel.setMemoization(subset)
        firstEval = object_cmi_dense_py_kernel.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_cmi_dense_py_kernel.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_cmi_dense_py_kernel.marginalGain(subset, elem)
        fastGain = object_cmi_dense_py_kernel.marginalGainWithMemoization(subset, elem)
        # assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
        assert naiveGain == pytest.approx(simpleGain) and simpleGain == pytest.approx(fastGain), "Mismatch between naive, simple and fast margins"

    ######## Tests for CMI optimizers ################
    @pytest.mark.cmi_opt_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", optimizerCMITests, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_naive_lazy(self, object_cmi_dense_py_kernel):
        greedyListNaive = object_cmi_dense_py_kernel.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_cmi_dense_py_kernel.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.cmi_opt_regular
    @pytest.mark.parametrize("object_cmi_dense_py_kernel", optimizerCMITests, indirect=['object_cmi_dense_py_kernel'])
    def test_cmi_stochastic_lazierThanLazy(self, object_cmi_dense_py_kernel):
        greedyListStochastic = object_cmi_dense_py_kernel.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_cmi_dense_py_kernel.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"
    

    ######## Optimizers test for SetCoverCMI #####################
    @pytest.mark.sc_cmi_opt
    def test_sc_cmi_optimizer_naive_lazy(self, data_cmi_concepts):
        object_sccmi, _ = data_cmi_concepts
        greedyListNaive = object_sccmi.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_sccmi.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.sc_cmi_opt
    def test_sc_cmi_optimizer_stochastic_lazierThanLazy(self, data_cmi_concepts):
        object_sccmi, _ = data_cmi_concepts
        greedyListStochastic = object_sccmi.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_sccmi.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for SetCover CMI Function #######################
    @pytest.mark.sc_cmi_regular
    def test_sc_cmi_eval_groundset(self, data_cmi_concepts):
        object_sccmi, _ = data_cmi_concepts
        groundSet = object_sccmi.getEffectiveGroundSet()
        eval = object_sccmi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.sc_cmi_regular
    def test_sc_cmi_eval_emptyset(self, data_cmi_concepts):
        object_sccmi, _ = data_cmi_concepts
        eval = object_sccmi.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.sc_cmi_regular
    def test_sc_cmi_gain_on_empty(self, data_cmi_concepts):
        object_sccmi, set1 = data_cmi_concepts
        elem = random.sample(set1, 1)[0]
        testSet = set()
        evalEmpty = object_sccmi.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_sccmi.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_sccmi.marginalGain(set(), elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch for gain on empty set"
        #assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.sc_cmi_regular
    def test_sc_cmi_eval_evalfast(self, data_cmi_concepts):
        object_sccmi, set1 = data_cmi_concepts
        subset = set()
        for elem in set1:
            object_sccmi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_sccmi.evaluate(subset)
        fastEval = object_sccmi.evaluateWithMemoization(subset)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.sc_cmi_regular
    def test_sc_cmi_set_memoization(self, data_cmi_concepts):
        object_sccmi, set1 = data_cmi_concepts
        object_sccmi.setMemoization(set1)
        simpleEval = object_sccmi.evaluate(set1)
        fastEval = object_sccmi.evaluateWithMemoization(set1)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.sc_cmi_regular
    def test_sc_cmi_gain(self, data_cmi_concepts):
        object_sccmi, set1 = data_cmi_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_sccmi.setMemoization(subset)
        firstEval = object_sccmi.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_sccmi.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_sccmi.marginalGain(subset, elem)
        fastGain = object_sccmi.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
        #assert naiveGain == simpleGain and simpleGain == fastGain, "Mismatch between naive, simple and fast margins"
    
    ######## Optimizers test for ProbabilisticSetCoverCMI #####################
    @pytest.mark.psc_cmi_opt
    def test_psc_cmi_optimizer_naive_lazy(self, data_cmi_prob_concepts):
        object_psccmi, _ = data_cmi_prob_concepts
        greedyListNaive = object_psccmi.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazy = object_psccmi.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        naiveGains = [x[1] for x in greedyListNaive]
        lazyGains = [x[1] for x in greedyListLazy]
        assert naiveGains == pytest.approx(lazyGains), "Mismatch between naiveGreedy and lazyGreedy"
    
    @pytest.mark.psc_cmi_opt
    def test_psc_cmi_optimizer_stochastic_lazierThanLazy(self, data_cmi_prob_concepts):
        object_psccmi, _ = data_cmi_prob_concepts
        greedyListStochastic = object_psccmi.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        greedyListLazierThanLazy = object_psccmi.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        stochasticGains = [x[1] for x in greedyListStochastic]
        lazierThanLazyGains = [x[1] for x in greedyListLazierThanLazy]
        assert stochasticGains == pytest.approx(lazierThanLazyGains), "Mismatch between stochasticGreedy and lazierThanLazyGreedy"

    ############ 6 regular tests for ProbabilisticSetCover CMI Function #######################
    @pytest.mark.psc_cmi_regular
    def test_psc_cmi_eval_groundset(self, data_cmi_prob_concepts):
        object_psccmi, _ = data_cmi_prob_concepts
        groundSet = object_psccmi.getEffectiveGroundSet()
        eval = object_psccmi.evaluate(groundSet)
        assert eval >= 0 and not math.isnan(eval) and not math.isinf(eval), "Eval on groundset is not >= 0 or is NAN or is INF"

    @pytest.mark.psc_cmi_regular
    def test_psc_cmi_eval_emptyset(self, data_cmi_prob_concepts):
        object_psccmi, _ = data_cmi_prob_concepts
        eval = object_psccmi.evaluate(set())
        assert eval == 0, "Eval on empty set is not = 0"
    
    @pytest.mark.psc_cmi_regular
    def test_psc_cmi_gain_on_empty(self, data_cmi_prob_concepts):
        object_psccmi, set1 = data_cmi_prob_concepts
        elem = random.sample(set1, 1)[0]
        testSet = set()
        evalEmpty = object_psccmi.evaluate(testSet)
        testSet.add(elem)
        evalSingleItem = object_psccmi.evaluate(testSet)
        gain1 = evalSingleItem - evalEmpty
        gain2 = object_psccmi.marginalGain(set(), elem)
        assert math.isclose(gain1, gain2, rel_tol=1e-05), "Mismatch for gain on empty set"
        #assert gain1 == gain2, "Mismatch for gain on empty set"

    @pytest.mark.psc_cmi_regular
    def test_psc_cmi_eval_evalfast(self, data_cmi_prob_concepts):
        object_psccmi, set1 = data_cmi_prob_concepts
        subset = set()
        for elem in set1:
            object_psccmi.updateMemoization(subset, elem)
            subset.add(elem)
        simpleEval = object_psccmi.evaluate(subset)
        fastEval = object_psccmi.evaluateWithMemoization(subset)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after incremental addition"

    @pytest.mark.psc_cmi_regular
    def test_psc_cmi_set_memoization(self, data_cmi_prob_concepts):
        object_psccmi, set1 = data_cmi_prob_concepts
        object_psccmi.setMemoization(set1)
        simpleEval = object_psccmi.evaluate(set1)
        fastEval = object_psccmi.evaluateWithMemoization(set1)
        #assert math.isclose(simpleEval, fastEval, rel_tol=1e-05), "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"
        assert simpleEval == fastEval, "Mismatch between evaluate() and evaluateWithMemoization after setMemoization"

    @pytest.mark.psc_cmi_regular
    def test_psc_cmi_gain(self, data_cmi_prob_concepts):
        object_psccmi, set1 = data_cmi_prob_concepts
        elems = random.sample(set1, num_random)
        subset = set(elems[:-1])
        elem = elems[-1]
        object_psccmi.setMemoization(subset)
        firstEval = object_psccmi.evaluateWithMemoization(subset)
        subset.add(elem)
        naiveGain = object_psccmi.evaluate(subset) - firstEval
        subset.remove(elem)
        simpleGain = object_psccmi.marginalGain(subset, elem)
        fastGain = object_psccmi.marginalGainWithMemoization(subset, elem)
        assert math.isclose(naiveGain, simpleGain, rel_tol=1e-05) and math.isclose(simpleGain, fastGain, rel_tol=1e-05), "Mismatch between naive, simple and fast margins"
        #assert naiveGain == simpleGain and simpleGain == fastGain, "Mismatch between naive, simple and fast margins"