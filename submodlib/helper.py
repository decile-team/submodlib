import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from sklearn.cluster import Birch #https://scikit-learn.org/stable/modules/clustering.html#birch


def create_kernel(X, mode, metric, num_neigh=-1, n_jobs=1, X_master=None):
    
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



def create_cluster(X, metric, num_cluster=None):#Here, metric only controls the similarity scores in sim-kernel of each cluster. 
                                                #Clustering always happens using Euclidean Distance
    obj = Birch(n_clusters=num_cluster)
    obj = obj.fit(X)
    lab = obj.predict(X).tolist()
    if num_cluster==None:
        num_cluster=len(obj.subcluster_labels_)
    
    l_cluster= [set() for _ in range(num_cluster)]
    l_ind = [0]*np.shape(X)[0]
    l_count = [0]*num_cluster
    
    for i, el in enumerate(lab):#For any cluster ID (el), smallest datapoint (i) is filled first
                                #Therefore, the set l_cluster will always be sorted
        l_cluster[el].add(i)
        l_ind[i]=l_count[el]
        l_count[el]=l_count[el]+1
        
    l_kernel =[]
    for el in l_cluster:
        k = len(el)
        l_kernel.append(np.zeros((k,k)))
    
    M=None
    if metric=="euclidean":
        D = euclidean_distances(X)
        gamma = 1/np.shape(X)[1]
        M = np.exp(-D * gamma) #Obtaining Similarity from distance
    else:
        if metric=="cosine":
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
        