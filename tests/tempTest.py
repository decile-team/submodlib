from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random
import time
import numpy as np

def test():

    num_clusters = 3
    cluster_std_dev = 1
    num_samples = 8
    num_set = 3
    budget = 4

    points, cluster_ids, centers = make_blobs(n_samples=num_samples, centers=num_clusters, n_features=2, cluster_std=cluster_std_dev, center_box=(0,100), return_centers=True, random_state=4)
    data = list(map(tuple, points))
    xs = [x[0] for x in data]
    ys = [x[1] for x in data]

    # get num_set data points belonging to cluster#1
    
    random.seed(1)
    cluster1Indices = [index for index, val in enumerate(cluster_ids) if val == 1]
    subset1 = random.sample(cluster1Indices, num_set)
    subset1xs = [xs[x] for x in subset1]
    subset1ys = [ys[x] for x in subset1]
    # plt.scatter(xs, ys, s=25, color='black', label="Images")
    # plt.scatter(subset1xs, subset1ys, s=25, color='red', label="Subset1")
    # plt.show()
    set1 = set(subset1[:-1])

    # get num_set data points belonging to different clusters
    subset2 = []
    for i in range(num_set):
        #find the index of first point that belongs to cluster i
        diverse_index = cluster_ids.tolist().index(i)
        subset2.append(diverse_index)
    subset2xs = [xs[x] for x in subset2]
    subset2ys = [ys[x] for x in subset2]
    # plt.scatter(xs, ys, s=25, color='black', label="Images")
    # plt.scatter(subset2xs, subset2ys, s=25, color='red', label="Subset2")
    # plt.show()
    set2 = set(subset2[:-1])
    
    dataArray = np.array(data)

    from submodlib.functions.facilityLocation import FacilityLocationFunction

    # start = time.process_time()
    obj5 = FacilityLocationFunction(n=num_samples, data=dataArray, mode="clustered", metric="euclidean", num_clusters=num_clusters)
    # print(f"Time taken by instantiation = {time.process_time() - start}")
    print(f"Subset 1's FL value = {obj5.evaluate(set1)}")
    print(f"Subset 2's FL value = {obj5.evaluate(set2)}")
    print(f"Gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj5.marginalGain(set1, subset1[-1])}")
    print(f"Gain of adding another point ({subset2[-1]}) of different cluster to {set1} = {obj5.marginalGain(set1, subset2[-1])}")
    obj5.setMemoization(set1)
    print(f"Subset 1's Fast FL value = {obj5.evaluateWithMemoization(set1)}")
    print(f"Fast gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj5.marginalGainWithMemoization(set1, subset1[-1])}")
    # start = time.process_time()
    greedyList = obj5.maximize(budget=budget,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(f"Greedy vector: {greedyList}")
    # print(f"Time taken by maximization = {time.process_time() - start}")
    # greedyXs = [xs[x[0]] for x in greedyList]
    # greedyYs = [ys[x[0]] for x in greedyList]
    # plt.scatter(xs, ys, s=25, color='black', label="Images")
    # plt.scatter(greedyXs, greedyYs, s=25, color='blue', label="Greedy Set")

    from submodlib import ClusteredFunction

    obj7 = ClusteredFunction(n=num_samples, mode="multi", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_clusters)
    # print(f"Time taken by instantiation = {time.process_time() - start}")
    print(f"Subset 1's FL value = {obj7.evaluate(set1)}")
    print(f"Subset 2's FL value = {obj7.evaluate(set2)}")
    print(f"Gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj7.marginalGain(set1, subset1[-1])}")
    print(f"Gain of adding another point ({subset2[-1]}) of different cluster to {set1} = {obj7.marginalGain(set1, subset2[-1])}")
    obj7.setMemoization(set1)
    print(f"Subset 1's Fast FL value = {obj7.evaluateWithMemoization(set1)}")
    print(f"Fast gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj7.marginalGainWithMemoization(set1, subset1[-1])}")
    # start = time.process_time()
    greedyList = obj7.maximize(budget=budget,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(f"Greedy vector: {greedyList}")
    # print(f"Time taken by maximization = {time.process_time() - start}")
    # greedyXs = [xs[x[0]] for x in greedyList]
    # greedyYs = [ys[x[0]] for x in greedyList]
    # plt.scatter(xs, ys, s=25, color='black', label="Images")
    # plt.scatter(greedyXs, greedyYs, s=25, color='blue', label="Greedy Set")



    # start = time.process_time()
    # obj9 = ClusteredFunction(n=num_samples, mode="single", f_name='FacilityLocation', metric='euclidean', data=dataArray, num_clusters=num_clusters)
    # print(f"Time taken by instantiation = {time.process_time() - start}")
    # print(f"Subset 1's FL value = {obj9.evaluate(set1)}")
    # print(f"Subset 2's FL value = {obj9.evaluate(set2)}")
    # print(f"Gain of adding another point ({subset1[-1]}) of same cluster to {set1} = {obj9.marginalGain(set1, subset1[-1])}")
    # print(f"Gain of adding another point ({subset2[-1]}) of different cluster to {set1} = {obj9.marginalGain(set1, subset2[-1])}")
    # start = time.process_time()
    # greedyList = obj9.maximize(budget=budget,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    # print(f"Time taken by maximization = {time.process_time() - start}")
    # print("Output of maximization", greedyList)
    # greedyXs = [xs[x[0]] for x in greedyList]
    # greedyYs = [ys[x[0]] for x in greedyList]
    # # plt.scatter(xs, ys, s=25, color='black', label="Images")
    # # plt.scatter(greedyXs, greedyYs, s=25, color='blue', label="Greedy Set")
    # # plt.show()

if __name__ == '__main__':
    test()