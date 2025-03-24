import torch
import torch.nn.functional as F
from sklearn.cluster import Birch
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
import pickle
import time
import os
import numpy as np
from typing import List, Dict, Union
from math import sqrt

# Define type aliases for clarity
Vector = List[float]
Matrix = List[Vector]
Set = List[int]  # Considering integer elements for simplicity

def cos_sim_square(A):
    similarity = torch.matmul(A, A.t())

    square_mag = torch.diag(similarity)

    inv_square_mag = 1 / square_mag
    inv_square_mag[torch.isinf(inv_square_mag)] = 0

    inv_mag = torch.sqrt(inv_square_mag)

    cosine = similarity * inv_mag
    cosine = cosine.t() * inv_mag
    return cosine

def cos_sim_rectangle(A, B):
    num = torch.matmul(A, B.t())
    p1 = torch.sqrt(torch.sum(A**2, dim=1)).unsqueeze(1)
    p2 = torch.sqrt(torch.sum(B**2, dim=1)).unsqueeze(0)
    return num / (p1 * p2)

def create_sparse_kernel(X, metric, num_neigh, n_jobs=1, method="sklearn"):
    if num_neigh > X.shape[0]:
        raise Exception("ERROR: num of neighbors can't be more than the number of datapoints")
    dense = None
    dense = create_kernel_dense_sklearn(X, metric)
    dense_ = None
    if num_neigh == -1:
        num_neigh = X.shape[0]  # default is the total number of datapoints

    # Assuming X is a PyTorch tensor
    X_np = X.numpy()

    # Use PyTorch functions for the nearest neighbors search
    if metric == 'euclidean':
      distances = torch.cdist(X, X, p=2)  # Euclidean distance
    elif metric == 'cosine':
      distances = 1 - torch.nn.functional.cosine_similarity(X, X, dim=1)  # Cosine similarity as distance

    # Exclude the distance to oneself (diagonal elements)
    distances.fill_diagonal_(float('inf'))

    # Find the indices of the k-nearest neighbors using torch.topk
    _, ind = torch.topk(distances, k=num_neigh, largest=False)

    # ind_l = [(index[0], x.item()) for index, x in torch.ndenumerate(ind)]
        # Convert indices to row and col lists
    row = []
    col = []
    for i, indices_row in enumerate(ind):
        for j in indices_row:
            row.append(i)
            col.append(j.item())

    mat = torch.zeros_like(distances)
    mat[row, col] = 1
    dense_ = dense * mat  # Only retain similarity of nearest neighbors
    sparse_coo = torch.sparse_coo_tensor(torch.tensor([row, col]), mat[row, col], dense.size())
    # Convert the COO tensor to CSR format
    sparse_csr = sparse_coo.coalesce()
    return sparse_csr
    # pass


def create_kernel_dense(X, metric, method="sklearn"):
    dense = None
    if method == "sklearn":
        dense = create_kernel_dense_sklearn(X, metric)
    else:
        raise Exception("For creating dense kernel, only 'sklearn' method is supported")
    return dense

def create_kernel_dense_sklearn(X, metric, X_rep=None, batch=0):
    dense = None
    D = None
    batch_size = batch
    if metric == "euclidean":
        if X_rep is None:
            # print(X.shape)
            # Process data in batches for torch.cdist
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device="cuda")
                # print(X_batch.shape)
                D_batch = torch.cdist(X_batch, X, p=2).to(device="cuda")
                gamma = 1 / X.shape[1]
                dense_batch = torch.exp(-D_batch * gamma).to(device="cuda")
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])
        else:
            # Process data in batches for torch.cdist
            for i in range(0, len(X_rep), batch_size):
                X_rep_batch = X_rep[i:i+batch_size].to(device="cuda")
                D_batch = torch.cdist(X_rep_batch, X).to(device="cuda")
                gamma = 1 / X.shape[1]
                dense_batch = torch.exp(-D_batch * gamma).to(device="cuda")
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])

    elif metric == "cosine":
        if X_rep is None:
            # Process data in batches for torch.nn.functional.cosine_similarity
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device="cuda")
                dense_batch = torch.nn.functional.cosine_similarity(X_batch.unsqueeze(1), X.unsqueeze(0), dim=2)
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])
        else:
            # Process data in batches for torch.nn.functional.cosine_similarity
            for i in range(0, len(X_rep), batch_size):
                X_rep_batch = X_rep[i:i+batch_size].to(device="cuda")
                dense_batch = torch.nn.functional.cosine_similarity(X_rep_batch, X, dim=1)
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])

    elif metric == "dot":
        if X_rep is None:
            # Process data in batches for torch.matmul
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size].to(device="cuda")
                dense_batch = torch.matmul(X_batch, X.t())
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])
        else:
            # Process data in batches for torch.matmul
            for i in range(0, len(X_rep), batch_size):
                X_rep_batch = X_rep[i:i+batch_size].to(device="cuda")
                dense_batch = torch.matmul(X_rep_batch, X.t())
                # Accumulate results from batches
                if dense is None:
                    dense = dense_batch
                else:
                    dense = torch.cat([dense, dense_batch])

    else:
        raise Exception("ERROR: unsupported metric for this method of kernel creation")

    if X_rep is not None:
        assert dense.shape == (X_rep.shape[0], X.shape[0])
    else:
        assert dense.shape == (X.shape[0], X.shape[0])

    torch.cuda.empty_cache()
    return dense

def create_cluster_kernels(X, metric, cluster_lab=None, num_cluster=None, onlyClusters=False):
    lab = []
    if cluster_lab is None:
        obj = Birch(n_clusters=num_cluster)
        obj.fit(X)
        lab = obj.predict(X).tolist()
        if num_cluster is None:
            num_cluster = len(obj.subcluster_labels_)
    else:
        if num_cluster is None:
            raise Exception("ERROR: num_cluster needs to be specified if cluster_lab is provided")
        lab = cluster_lab
    
    l_cluster = [set() for _ in range(num_cluster)]
    l_ind = [0] * X.shape[0]
    l_count = [0] * num_cluster
    
    for i, el in enumerate(lab):
        l_cluster[el].add(i)
        l_ind[i] = l_count[el]
        l_count[el] = l_count[el] + 1

    if onlyClusters:
        return l_cluster, None, None
        
    l_kernel = []
    for el in l_cluster: 
        k = len(el)
        l_kernel.append(torch.zeros((k, k)))  # placeholder matrices of suitable size
    
    M = None
    if metric == "euclidean":
        D = torch.cdist(X, X)
        gamma = 1 / X.shape[1]
        M = torch.exp(-D * gamma)  # similarity from distance
    elif metric == "cosine":
        M = F.cosine_similarity(X, X, dim=1)
        M = M.unsqueeze(0)  # converting to 2D for compatibility
    else:
        raise Exception("ERROR: unsupported metric")
    
    # Create kernel for each cluster using the bigger kernel
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if lab[i] == lab[j]:
                c_ID = lab[i]
                ii = l_ind[i]
                jj = l_ind[j]
                l_kernel[c_ID][ii, jj] = M[i, j]
            
    return l_cluster, l_kernel, l_ind

def create_kernel(X, metric, mode="dense", num_neigh=-1, n_jobs=1, X_rep=None, method="sklearn"):

    if X_rep is not None:
        assert X_rep.shape[1] == X.shape[1]

    if mode == "dense":
        dense = None
        dense = globals()['create_kernel_dense_'+method](X, metric, X_rep)
        return torch.tensor(dense)

    elif mode == "sparse":
        if X_rep is not None:
            raise Exception("Sparse mode is not supported for separate X_rep")
        return create_sparse_kernel(X, metric, num_neigh, n_jobs, method)

    else:
        raise Exception("ERROR: unsupported mode")



# Euclidean similarity function
def euclidean_similarity(a: Vector, b: Vector) -> float:
    return np.linalg.norm(np.array(a) - np.array(b))

# Cosine similarity function
def cosine_similarity(a: Vector, b: Vector) -> float:
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

# Dot product function
def dot_prod(a: Vector, b: Vector) -> float:
    return np.dot(a, b)

# Create kernel function for non-square kernel
def create_kernel_NS(X_ground: Matrix, X_master: Matrix, metric: str = "euclidean") -> Matrix:
    n_ground = len(X_ground)
    n_master = len(X_master)
    k_dense = [[0] * n_ground for _ in range(n_master)]

    for r in range(n_master):
        for c in range(n_ground):
            if metric == "euclidean":
                k_dense[r][c] = euclidean_similarity(X_master[r], X_ground[c])
            elif metric == "cosine":
                k_dense[r][c] = cosine_similarity(X_master[r], X_ground[c])
            elif metric == "dot":
                k_dense[r][c] = dot_prod(X_master[r], X_ground[c])
            else:
                raise ValueError("Unsupported metric for kernel computation in Python")
    return k_dense

# Create square kernel function
def create_square_kernel_dense(X_ground: Matrix, metric: str = "euclidean") -> Matrix:
    n_ground = len(X_ground)
    k_dense = [[0] * n_ground for _ in range(n_ground)]

    if metric == "euclidean":
        for r in range(n_ground):
            k_dense[r][r] = 1.0
            for c in range(r + 1, n_ground):
                sim = euclidean_similarity(X_ground[r], X_ground[c])
                k_dense[r][c] = sim
                k_dense[c][r] = sim
    elif metric == "cosine":
        for r in range(n_ground):
            a_norm = sqrt(dot_prod(X_ground[r], X_ground[r]))
            k_dense[r][r] = 1.0
            for c in range(r + 1, n_ground):
                sim = dot_prod(X_ground[r], X_ground[c])
                b_norm = sqrt(dot_prod(X_ground[c], X_ground[c]))
                sim = sim / (a_norm * b_norm) if a_norm * b_norm > 0 else 0
                k_dense[r][c] = sim
                k_dense[c][r] = sim
    elif metric == "dot":
        for r in range(n_ground):
            for c in range(r, n_ground):
                sim = dot_prod(X_ground[r], X_ground[c])
                k_dense[r][c] = sim
                k_dense[c][r] = sim
    else:
        raise ValueError("Unsupported metric for kernel computation in Python")
    return k_dense

# Set intersection function
def set_intersection(a: Set, b: Set) -> Set:
    return list(set(a) & set(b))  # Converting set intersection to list for better compatibility

