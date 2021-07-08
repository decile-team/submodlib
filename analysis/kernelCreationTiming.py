# kernelCreationTiming.py
# Author: Vishal Kaushal
# Run as 'python kernelCreationTiming.py' to compare performance of different methods of creating kernel in python
# Uses naive time.time, runs a method num_executions times

from sklearn.datasets import make_blobs
import random
import numpy as np
#import submodlib_cpp as subcp
import submodlib.helper as helper
import time
import csv
from numba import threading_layer

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

cluster_std_dev = 2
num_executions = 10
num_places = 6
num_features = 1024

params = [(50, 5), (100, 10), (200, 10), (500, 10), (1000, 10), (5000, 10)]
#params = [(50, 5), (100, 10)]
first = True

for param in params:
    results_csv = [["Num_Samples", "Function", "Time"]]
    print("Parameters: ", param)
    #prepare data to be used in the analysis
    num_samples = param[0]
    num_clusters = param[1]

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=num_features, cluster_std=cluster_std_dev, return_centers=True, random_state=4)
    data = list(map(tuple, points))
    dataArray = np.array(data)

    if first == True:
        print("Pre compiling np_numba and fastdist functions")
        helper.create_kernel_dense_np_numba(dataArray, 'euclidean')
        print("Threading layer chosen: %s" % threading_layer())
        helper.create_kernel_dense_fastdist(dataArray, 'euclidean')
        print("Threading layer chosen: %s" % threading_layer())
        first = False

    for method in method_dictionary:
        print("Method: ", method)
        row=[num_samples, method]
        totalTime = 0
        for i in range(num_executions):
            func = getattr(helper, method_dictionary[method])
            start = time.time()
            kernel = func(dataArray, "euclidean")
            stop = time.time()
            currentTime = stop - start
            totalTime += currentTime
            row.append(round(currentTime,num_places))
        actualTime = totalTime/num_executions
        row.append(round(actualTime, num_places))
        results_csv.append(row)

    with open("kernel_creation_timings_" + str(num_samples) + ".csv", "w") as f:
        writer = csv.writer(f)
        for result in results_csv:
            writer.writerow(result)
