import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse


def create_kernel(X, X_master=None, mode, metric, num_neigh=-1, n_jobs=1):
    
    if X_master!=None and mode=="sparse":
        raise Exception("ERROR: mode can't be sparse if X_master isn't None")

    if num_neigh==-1 and X_master==None:
        num_neigh=np.shape(X)[0] #default is total no of datapoints

    if num_neigh>np.shape(X)[0]:
        raise Exception("ERROR: num of neighbors can't be more than no of datapoints")
    
    if mode in ['dense', 'sparse']:
        dense=None
        D=None
        if metric=="euclidean":
            if X_master==None:
                D = euclidean_distances(X)
            else:
                D = euclidean_distances(X_master, X)
            gamma = 1/np.shape(X)[1]
            dense = np.exp(-D * gamma) #Obtaining Similarity from distance
        else:
            if metric=="cosine":
                #D = cosine_distances(X) 
                if X_master==None:
                    dense = cosine_similarity(X)
                else:
                    dense = cosine_similarity(X_master, X)
            else:
                raise Exception("ERROR: unsupported metric")
        
        #nbrs = NearestNeighbors(n_neighbors=2, metric='precomputed', n_jobs=n_jobs).fit(D)
        dense_ = None
        if X_master==None:
            nbrs = NearestNeighbors(n_neighbors=num_neigh, metric=metric, n_jobs=n_jobs).fit(X)
            _, ind = nbrs.kneighbors(X)
            ind_l = [(index[0],x) for index, x in np.ndenumerate(ind)]
            row, col = zip(*ind_l)
            mat = np.zeros(np.shape(dense))
            mat[row, col]=1
            dense_ = dense*mat #Only retain similarity of nearest neighbours
        else:
            dense_ = dense

        if mode=='dense': 
            if num_neigh!=-1:       
                return num_neigh, dense_
            else:
                return dense_
        else:
            sparse_csr = sparse.csr_matrix(dense_)
            return num_neigh, sparse_csr
      
    else:
        raise Exception("ERROR: unsupported mode")