import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


def create_kernel(X, mode, metric, num_neigh=0, n_jobs=1):
    if mode=='sparse' and num_neigh==0:
        print("ERROR: num of neighbors not provided")
        return None
    if mode=='sparse' and num_neigh>np.shape(X)[0]:
        print("ERROR: num of neighbors can't be more than no of datapoints")
        return None
    
    if mode in ['dense', 'sparse']:
        dense=None
        D=None
        if metric=="euclidean":
            D = euclidean_distances(X) 
            gamma = 1/np.shape(X)[1]
            dense = np.exp(-D * gamma) #Obtaining Similarity from distance
        else:
            if metric=="cosine":
                #D = cosine_distances(X) 
                dense = cosine_similarity(X) 
            else:
                print("ERROR: unsupported metric")
                return None
        
        if mode=='dense':
            return dense.tolist() #Return kernel as list of list
        else:
            #nbrs = NearestNeighbors(n_neighbors=2, metric='precomputed', n_jobs=n_jobs).fit(D)
            nbrs = NearestNeighbors(n_neighbors=num_neigh, metric=metric, n_jobs=n_jobs).fit(X)
            _, ind = nbrs.kneighbors(X)
            ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
            row, col = zip(*ind_l)
            mat = np.zeros(np.shape(dense))
            mat[row, col]=1
            dense_ = dense*mat #Only retain similarity of nearest neighbours
            sparse_csr = sparse.csr_matrix(dense_)
            ds_csr = {}
            ds_csr['arr_val'] = m.data.tolist() #contains non-zero values in matrix (row major traversal)
            ds_csr['arr_count'] = m.indptr.tolist() #cumulitive count of non-zero elements upto but not including current row
            ds_csr['arr_col'] = m.indices.tolist() #contains col index corrosponding to non-zero values in arr_val
            
            return ds_csr 
      
    else:
        print("ERROR: unsupported mode")
        return None