from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib import ClusteredFunction
from memory_profiler import profile

@profile
def FL_sparse(data): #Case1 user only provides data and specifies
    obj = FacilityLocationFunction(n=43, data=data, mode="sparse", metric="euclidean", num_neigh=10)
    obj.maximize(10,'NaiveGreedy', False, False, False)

@profile
def FL_dense(data): #Case1 user only provides data and specifies
    obj = FacilityLocationFunction(n=43, data=data, mode="dense", metric="euclidean", num_neigh=10)
    obj.maximize(10,'NaiveGreedy', False, False, False)

@profile
def FL_case2(M): #Case2 user directly provides kernel
    obj = FacilityLocationFunction(n=43, sijs = M, num_neigh=10, seperateMaster=False)
    obj.maximize(10,'NaiveGreedy', False, False, False)
    

@profile
def FL_clustered_case1(data):#Case1 user only provides data
    obj = FacilityLocationFunction(n=43, data=data, mode="clustered", metric="euclidean", num_cluster=10)
    obj.maximize(10,'NaiveGreedy', False, False, False)
    
    
@profile
def FL_clustered_case2(data, lab):#Case2 user also provides cluster info along with data
    obj = FacilityLocationFunction(n=43, data=data, cluster_lab=lab, mode="clustered", metric="euclidean",num_cluster=10)
    obj.maximize(10,'NaiveGreedy', False, False, False)

#Similarly memory-analysis for other cases can be written
    
    

