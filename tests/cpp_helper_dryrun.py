import submodlib_cpp as subcp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from time import time

data = np.array([[0, 1, 3], [5, 1, 5], [10, 2, 6], [12,20,68]])

num_neigh=2
t = time()
CS = cosine_similarity(data) 
print("sklearn:", time()-t)
print(CS.tolist())

t = time()
c = subcp.create_kernel(data.tolist(), 'cosine', num_neigh)
val = c[0]
row = list(map(lambda arg: int(arg), c[1]))
col = list(map(lambda arg: int(arg), c[2]))
s = np.zeros((4,4))
s[row, col] = val
print("cpp:",time()-t) #Atleast 100 times faster
print(s)

#print(np.allclose(CS,s))


t = time()
ED = euclidean_distances(data) 
gamma = 1/np.shape(data)[1] 
ES = np.exp(-ED* gamma) #sklearn ground truth 
print("sklearn:", time()-t)
print(ES.tolist())

t = time()
c = subcp.create_kernel(data.tolist(), 'euclidean', num_neigh)
val = c[0]
row = list(map(lambda arg: int(arg), c[1]))
col = list(map(lambda arg: int(arg), c[2]))
s = np.zeros((4,4))
s[row, col] = val
print("cpp:",time()-t) #Atleast 10 times faster
print(s)

#print(np.allclose(ES,s))
