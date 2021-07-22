import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from sklearn.cluster import Birch #https://scikit-learn.org/stable/modules/clustering.html#birch
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
#from fastdist import fastdist
from numba import jit, config
import pickle
import time
import os

#from tqdm import tqdm
#from tqdm import trange

#config.THREADING_LAYER = 'default'

#TODO: https://stackoverflow.com/questions/13079563/how-does-condensed-distance-matrix-work-pdist
#Represent a dense kernel with upper triangular entries only to save memory
def cos_sim_square(A):
    # base similarity matrix (all dot products)
    similarity = np.dot(A, A.T)

    # squared magnitude of vectors
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

def cos_sim_rectangle(A, B):
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def euc_dis(A, B):
    M = A.shape[0]
    N = B.shape[0]
    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)
    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)

@jit(nopython=True, parallel=True)
def euc_dis_numba(A, B):
    M = A.shape[0]
    N = B.shape[0]
    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)
    D_squared = np.where(D_squared < 0.0, 0, D_squared)
    #zero_mask = np.less(D_squared, 0.0)
    #D_squared[zero_mask] = 0.0
    return np.sqrt(D_squared)

@jit(nopython=True, parallel=True)
def cos_sim_square_numba(A):
    # base similarity matrix (all dot products)
    similarity = np.dot(A, A.T)

    # squared magnitude of vectors
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

@jit(nopython=True, parallel=True)
def cos_sim_rectangle_numba(A, B):
    num=np.dot(A,B.T)
    p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)

def create_sparse_kernel(X, metric, num_neigh, n_jobs=1, method="sklearn"):
    if num_neigh>np.shape(X)[0]:
        raise Exception("ERROR: num of neighbors can't be more than no of datapoints")
    dense = None
    if method == "sklearn":
        dense = create_kernel_dense_sklearn(X, metric)
    elif method == "np_numba":
        dense = create_kernel_dense_np_numba(X, metric)
    else:
        raise Exception("For creating sparse kernel, only 'sklearn' and 'np_numba' methods are supported")
    #nbrs = NearestNeighbors(n_neighbors=2, metric='precomputed', n_jobs=n_jobs).fit(D)
    dense_ = None
    if num_neigh==-1:
        num_neigh=np.shape(X)[0] #default is total no of datapoints
    nbrs = NearestNeighbors(n_neighbors=num_neigh, metric=metric, n_jobs=n_jobs).fit(X)
    _, ind = nbrs.kneighbors(X)
    ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
    row, col = zip(*ind_l)   #unzipping
    mat = np.zeros(np.shape(dense))
    mat[row, col]=1
    dense_ = dense*mat #Only retain similarity of nearest neighbours
    sparse_csr = sparse.csr_matrix(dense_)
    return sparse_csr

# @jit(nopython=True, cache=True, parallel=True)
# def create_kernel_numba(X, metric, mode="dense", num_neigh=-1, n_jobs=1, X_rep=None):
#     #print(type(X_rep))
#     if type(X_rep)!=type(None) and mode=="sparse":
#         raise Exception("ERROR: sparse mode not allowed when using rectangular kernel")

#     if type(X_rep)!=type(None) and num_neigh!=-1:
#         raise Exception("ERROR: num_neigh can't be specified when using rectangular kernel")

#     if num_neigh==-1 and type(X_rep)==type(None):
#         num_neigh=np.shape(X)[0] #default is total no of datapoints

#     if num_neigh>np.shape(X)[0]:
#         raise Exception("ERROR: num of neighbors can't be more than no of datapoints")
    
#     if mode in ['dense', 'sparse']:
#         dense=None
#         D=None
#         if metric=="euclidean":
#             if type(X_rep)==type(None):
#                 D = euclidean_distances(X)
#             else:
#                 D = euclidean_distances(X_rep, X)
#             gamma = 1/np.shape(X)[1]
#             dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#         else:
#             if metric=="cosine":
#                 if type(X_rep)==type(None):
#                     dense = cosine_similarity(X)
#                 else:
#                     dense = cosine_similarity(X_rep, X)
#             else:
#                 raise Exception("ERROR: unsupported metric")
        
#         #nbrs = NearestNeighbors(n_neighbors=2, metric='precomputed', n_jobs=n_jobs).fit(D)
#         dense_ = None
#         if type(X_rep)==type(None):
#             nbrs = NearestNeighbors(n_neighbors=num_neigh, metric=metric, n_jobs=n_jobs).fit(X)
#             _, ind = nbrs.kneighbors(X)
#             ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
#             row, col = zip(*ind_l)
#             mat = np.zeros(np.shape(dense))
#             mat[row, col]=1
#             dense_ = dense*mat #Only retain similarity of nearest neighbours
#         else:
#             dense_ = dense

#         if mode=='dense': 
#             if num_neigh!=-1:       
#                 return num_neigh, dense_
#             else:
#                 return dense_ #num_neigh is not returned because its not a sensible value in case of rectangular kernel
#         else:
#             sparse_csr = sparse.csr_matrix(dense_)
#             return num_neigh, sparse_csr
      
#     else:
#         raise Exception("ERROR: unsupported mode")

#@jit(nopython=True, parallel=True)
def create_kernel_dense_other(X, metric, X_rep=None):
    D = None
    if metric == 'euclidean':
        D = pairwise_distances(X, Y=X_rep, metric='euclidean', squared=True)
        D = np.subtract(D.max(), D, out=D)
    elif metric == 'cosine':
        D = pairwise_distances(X, Y=X_rep, metric="cosine")
        D = np.subtract(1, D, out=D)
        D = np.square(D, out=D)
        D = np.subtract(1, D, out=D)
        D = np.subtract(1, D, out=D)
    else:
        raise Exception("Unsupported metric for this method of kernel creation")
    if type(X_rep) != type(None):
        assert(D.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(D.shape == (X.shape[0], X.shape[0]))
    return D


def create_kernel_dense_rowwise(X, metric, X_rep=None):
    if type(X_rep) != type(None):
        num_rows = X_rep.shape[0]
    else:
        num_rows = X.shape[0]
    tempFile = 'kernel'+str(time.time())
    with open(tempFile, 'ab') as f:
        if metric == "cosine":
            if type(X_rep) == type(None):
                #for i in tqdm(X):
                for i in X:
                    similarity = cosine_similarity(i.reshape(1, -1), X).flatten()
                    pickle.dump(similarity, f)
            else:
                #for i in tqdm(X_rep):
                for i in X_rep:
                    similarity = cosine_similarity(i.reshape(1, -1), X).flatten()
                    pickle.dump(similarity, f)
        elif metric == "euclidean":
            if type(X_rep) == type(None):
                #for i in tqdm(X):
                for i in X:
                    distance = euclidean_distances(i.reshape(1, -1), X).flatten()
                    pickle.dump(distance, f)
            else:
                #for i in tqdm(X_rep):
                for i in X_rep:
                    distance = euclidean_distances(i.reshape(1, -1), X).flatten()
                    pickle.dump(distance, f)
        elif metric == "dot":
            if type(X_rep) == type(None):
                #for i in tqdm(X):
                for i in X:
                    similarity = np.dot(i.reshape(1, -1), X.T).flatten()
                    pickle.dump(similarity, f)
            else:
                #for i in tqdm(X_rep):
                for i in X_rep:
                    similarity = np.dot(i.reshape(1, -1), X.T).flatten()
                    pickle.dump(similarity, f)
        else:
            raise Exception("Unsupported metric for this method of kernel creation")
    with open(tempFile, 'rb') as f:
        D = []
        #for i in trange(num_rows):
        for i in range(num_rows):
            D.append(pickle.load(f))
        D = np.array(D)
        if metric == "cosine" or metric == "dot":
            dense = D
        elif metric == "euclidean":
            gamma = 1/np.shape(X)[1]
            dense = np.exp(-D * gamma)
    os.remove(tempFile)
    assert(dense.shape == (num_rows, X.shape[0]))
    return dense

def create_kernel_dense_sklearn(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        if type(X_rep)==type(None):
            D = euclidean_distances(X)
        else:
            D = euclidean_distances(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        if type(X_rep)==type(None):
            dense = cosine_similarity(X)
        else:
            dense = cosine_similarity(X_rep, X)
    elif metric == "dot":
        if type(X_rep)==type(None):
            dense = np.matmul(X, X.T)
        else:
            dense = np.matmul(X_rep, X.T)
    else:
        raise Exception("ERROR: unsupported metric for this method of kernel creation")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

# @jit(nopython=True, cache=True, parallel=True)
# def create_kernel_dense_sklearn_numba(X, metric, X_rep=None):
#     dense=None
#     D=None
#     if metric=="euclidean":
#         if type(X_rep)==type(None):
#             D = euclidean_distances(X)
#         else:
#             D = euclidean_distances(X_rep, X)
#         gamma = 1/np.shape(X)[1]
#         dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#     else:
#         if metric=="cosine":
#             if type(X_rep)==type(None):
#                 dense = cosine_similarity(X)
#             else:
#                 dense = cosine_similarity(X_rep, X)
#         else:
#             raise Exception("ERROR: unsupported metric")
#     return dense

def create_kernel_dense_scipy(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        if type(X_rep)==type(None):
            D = distance_matrix(X, X)
        else:
            D = distance_matrix(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        if type(X_rep)==type(None):
            D = pdist(X, metric="cosine")
            dense = squareform(D, checks=False)
            dense = 1-dense
        else:
            dense = cdist(X_rep, X, metric="cosine")
            dense = 1-dense
    else:
        raise Exception("ERROR: unsupported metric")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

# @jit(nopython=True, cache=True, parallel=True)
# def create_kernel_dense_scipy_numba(X, metric, X_rep=None):
#     dense=None
#     D=None
#     if metric=="euclidean":
#         if type(X_rep)==type(None):
#             D = distance_matrix(X, X)
#         else:
#             D = distance_matrix(X_rep, X)
#         gamma = 1/np.shape(X)[1]
#         dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#     else:
#         if metric=="cosine":
#             if type(X_rep)==type(None):
#                 D = pdist(X, metric="cosine")
#                 dense = squareform(D, checks=False)
#                 dense = 1-dense
#             else:
#                 dense = cdist(X_rep, X, metric="cosine")
#                 dense = 1-dense
#         else:
#             raise Exception("ERROR: unsupported metric")
#     return dense

# def create_kernel_dense_fastdist(X, metric, X_rep=None):
#     dense=None
#     D=None
#     if metric=="euclidean":
#         if type(X_rep)==type(None):
#             D = fastdist.matrix_pairwise_distance(X, fastdist.euclidean, "euclidean", return_matrix=True)
#         else:
#             D = fastdist.matrix_to_matrix_distance(X_rep, X, fastdist.euclidean, "euclidean")
#         gamma = 1/np.shape(X)[1]
#         dense = np.exp(-D * gamma) #Obtaining Similarity from distance
#     elif metric=="cosine":
#         if type(X_rep)==type(None):
#             D = fastdist.matrix_pairwise_distance(X, fastdist.cosine, "cosine", return_matrix=True)
#         else:
#             D = fastdist.matrix_to_matrix_distance(X_rep, X, fastdist.cosine, "cosine")
#         #dense = 1-D
#         dense = D
#     else:
#         raise Exception("ERROR: unsupported metric")
#     if type(X_rep) != type(None):
#         assert(dense.shape == (X_rep.shape[0], X.shape[0]))
#     else:
#         assert(dense.shape == (X.shape[0], X.shape[0]))
#     return dense

def create_kernel_dense_np(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        if type(X_rep)==type(None):
            D = euc_dis(X, X)
        else:
            D = euc_dis(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        if type(X_rep)==type(None):
            dense = cos_sim_square(X)
        else:
            dense = cos_sim_rectangle(X_rep, X)
    elif metric=="dot":
        if type(X_rep)==type(None):
            dense = np.matmul(X, X.T)
        else:
            dense = np.matmul(X_rep, X.T)
    else:
        raise Exception("ERROR: unsupported metric")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

@jit(nopython=True, parallel=True)
def create_kernel_dense_np_numba(X, metric, X_rep=None):
    dense=None
    D=None
    if metric=="euclidean":
        D = euc_dis_numba(X, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        dense = cos_sim_square_numba(X)
    # elif metric=="dot":
    #     dense = np.matmul(X, X.T)
    else:
        raise Exception("ERROR: unsupported metric")
    assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

@jit(nopython=True, cache=True, parallel=True)
def create_kernel_dense_np_numba_rectangular(X, metric, X_rep):
    dense=None
    D=None
    if metric=="euclidean":
        D = euc_dis_numba(X_rep, X)
        gamma = 1/np.shape(X)[1]
        dense = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        dense = cos_sim_rectangle_numba(X_rep, X)
    elif metric=="dot":
        dense = np.matmul(X_rep, X.T)
    else:
        raise Exception("ERROR: unsupported metric for this method of kernel creation")
    if type(X_rep) != type(None):
        assert(dense.shape == (X_rep.shape[0], X.shape[0]))
    else:
        assert(dense.shape == (X.shape[0], X.shape[0]))
    return dense

def create_cluster_kernels(X, metric, cluster_lab=None, num_cluster=None, onlyClusters=False): #Here cluster_lab is a list which specifies custom cluster mapping of a datapoint to a cluster
    
    lab=[]
    if cluster_lab==None:
        obj = Birch(n_clusters=num_cluster) #https://scikit-learn.org/stable/modules/clustering.html#birch
        obj = obj.fit(X)
        lab = obj.predict(X).tolist()
        if num_cluster==None:
            num_cluster=len(obj.subcluster_labels_)
    else:
        if num_cluster==None:
            raise Exception("ERROR: num_cluster needs to be specified if cluster_lab is provided")
        lab=cluster_lab
    
    #print("Custer labels: ", lab)

    l_cluster= [set() for _ in range(num_cluster)]
    l_ind = [0]*np.shape(X)[0]
    l_count = [0]*num_cluster
    
    for i, el in enumerate(lab):#For any cluster ID (el), smallest datapoint (i) is filled first
                                #Therefore, the set l_cluster will always be sorted
        #print(f"{i} is in cluster {el}")
        l_cluster[el].add(i)
        l_ind[i]=l_count[el]
        l_count[el]=l_count[el]+1

    #print("l_cluster inside helper: ", l_cluster)

    if onlyClusters:
        return l_cluster, None, None
        
    l_kernel =[]
    for el in l_cluster: 
        k = len(el)
        l_kernel.append(np.zeros((k,k))) #putting placeholder matricies of suitable size
    
    M=None
    if metric=="euclidean":
        D = euclidean_distances(X)
        gamma = 1/np.shape(X)[1]
        M = np.exp(-D * gamma) #Obtaining Similarity from distance
    elif metric=="cosine":
        M = cosine_similarity(X)
    else:
        raise Exception("ERROR: unsupported metric")
    
    #Create kernel for each cluster using the bigger kernel
    for ind, val in np.ndenumerate(M): 
        if lab[ind[0]]==lab[ind[1]]:#if a pair of datapoints is in same cluster then update the kernel corrosponding to that cluster 
            c_ID = lab[ind[0]]
            i=l_ind[ind[0]]
            j=l_ind[ind[1]]
            l_kernel[c_ID][i,j]=val
            
    return l_cluster, l_kernel, l_ind

def create_kernel(X, metric, mode="dense", num_neigh=-1, n_jobs=1, X_rep=None, method="sklearn"):
    if type(X_rep) != type(None):
        assert(X_rep.shape[1] == X.shape[1])
    if mode == "dense":
        dense = None
        if method == "np_numba" and type(X_rep) != type(None):
            dense = create_kernel_dense_np_numba_rectangular(X, metric, X_rep)
        else:
            dense = globals()['create_kernel_dense_'+method](X, metric, X_rep)
        return dense
    elif mode == "sparse":
        if type(X_rep) != type(None):
            raise Exception("Sparse mode is not supported for separate X_rep")
        return create_sparse_kernel(X, metric, num_neigh, n_jobs, method)
    else:
        raise Exception("ERROR: unsupported mode")