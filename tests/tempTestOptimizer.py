from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random
import time
import numpy as np

def test():

    num_clusters = 3 #10 #100 #3
    cluster_std_dev = 1 #4 #4 #1
    num_samples = 9 #500 #5000 #9
    budget = 4 #10 #10 #4

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
    data = list(map(tuple, points))
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]
    #plt.scatter(xs, ys, s=25, color='black', label="Images")
    #plt.show()

    dataArray = np.array(data)

    from submodlib.functions.facilityLocation import FacilityLocationFunction
    obj = FacilityLocationFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
    print("Testing FacilityLocation's maximize")

    # from submodlib.functions.disparitySum import DisparitySumFunction
    # obj = DisparitySumFunction(n=num_samples, mode="dense", data=dataArray, metric="euclidean")
    # print("Testing DisparitySum's maximize")

    #greedyList = obj.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    greedyList = obj.maximize(budget=budget, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    #greedyList = obj.maximize(budget=budget, optimizer='StochasticGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    #greedyList = obj.maximize(budget=budget, optimizer='LazierThanLazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(f"Greedy vector: {greedyList}")
    greedyXs = [xs[x[0]] for x in greedyList]
    greedyYs = [ys[x[0]] for x in greedyList]
    #plt.scatter(xs, ys, s=25, color='black', label="Images")
    #plt.scatter(greedyXs, greedyYs, s=25, color='blue', label="Greedy Set")
    #plt.show()

if __name__ == '__main__':
    test()