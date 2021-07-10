import numpy as np
#import submodlib_cpp as subcp
import submodlib.helper as helper

groundData =np.array( [(4.5,13.5), (5,13.5), (5.5,13.5)] )

methods = ["sklearn", "fastdist", "scipy", "rowwise", "np", "np_numba"]

for method in methods:
    print("\n***Kernel from ", method)
    kernel = helper.create_kernel(groundData, "euclidean", method=method)
    print(kernel)

for method in methods:
    print("\n***Kernel from ", method)
    kernel = helper.create_kernel(groundData, "cosine", method=method)
    print(kernel)