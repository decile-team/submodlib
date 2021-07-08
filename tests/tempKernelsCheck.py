import numpy as np
#import submodlib_cpp as subcp
import submodlib.helper as helper

method_dictionary = {"np_numba": "create_kernel_dense_np_numba",
                     "np": "create_kernel_dense_np",
                     "fastdist": "create_kernel_dense_fastdist",
                     #"scipy_numba": "create_kernel_dense_scipy_numba",
                     "scipy": "create_kernel_dense_scipy",
                     #"sklearn_numba": "create_kernel_dense_sklearn_numba",
                     "sklearn": "create_kernel_dense_sklearn",
                     #"current_numba": "create_kernel_numba",
                     "current": "create_kernel"
}

groundData =np.array( [(4.5,13.5), (5,13.5), (5.5,13.5)] )

for method in method_dictionary:
    print("\n***Kernel from ", method)
    func = getattr(helper, method_dictionary[method])
    kernel = func(groundData, "euclidean")
    print(kernel)