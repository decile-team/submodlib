#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "LazyGreedyOptimizer.h"

LazyGreedyOptimizer::LazyGreedyOptimizer() {}

bool LazyGreedyOptimizer::equals(double val1, double val2, double eps) {
  if (abs(val1 - val2) < eps)
    return true;
  else {
    return false;
  }
}

std::vector<std::pair<ll, float>> LazyGreedyOptimizer::maximize(
    SetFunction &f_obj, ll budget, bool stopIfZeroGain = false,
    bool stopIfNegativeGain = false, bool verbose = false) {
    //TODO: take care of handling equal guys later
	//TODO: take care of different sizes of each items - becomes a candidate only if best and within budget, cost sensitive selection
    std::vector<std::pair<ll, float>> greedyVector;
    greedyVector.reserve(budget);
    std::unordered_set<ll> greedySet;
    greedySet.reserve(budget);
    ll rem_budget = budget;
    std::unordered_set<ll> groundSet = f_obj.getEffectiveGroundSet();
    if (verbose) {
        std::cout << "Ground set:" << std::endl;
        for (int i : groundSet) {
            std::cout << i << " ";
        }
        std::cout << "\n";
        std::cout << "Num elements in groundset = " << groundSet.size()
                  << std::endl;
        std::cout << "Starting the lazy greedy algorithm\n";
        std::cout << "Initial greedy set:" << std::endl;
        for (int i : greedySet) {
            std::cout << i << " ";
        }
        std::cout << "\n";
    }
    f_obj.clearMemoization();
    // initialize priority queue:
    std::priority_queue<std::pair<float, ll>> maxHeap;
    // for each element in the ground set
    for (auto elem : groundSet) {
        // store <elem, marginalGainWithMemoization(greedySet, elem)> in
        // priority-queue (max-heap)
        maxHeap.push(std::pair<float, ll>(
            f_obj.marginalGainWithMemoization(greedySet, elem), elem));
    }
    while (rem_budget > 0) {
        std::pair<float, ll> currentMax = maxHeap.top();
        maxHeap.pop();
        float newMaxBound =
            f_obj.marginalGainWithMemoization(greedySet, currentMax.second);
        if (newMaxBound > maxHeap.top().first) {
            // add currentMax.first to greedy set after checking stop conditions
            if ((newMaxBound < 0 && stopIfNegativeGain) ||
                (equals(newMaxBound, 0, 1e-5) && stopIfZeroGain)) {
                break;
            } else {
                f_obj.updateMemoization(greedySet, currentMax.second);
                greedySet.insert(
                    currentMax
                        .second);  // greedily insert the best datapoint index
                                   // of current iteration of while loop
                greedyVector.push_back(
                    std::pair<ll, float>(currentMax.second, newMaxBound));
                rem_budget -= 1;
                if (verbose) {
                    std::cout << "Added element " << currentMax.second
                              << " and the gain is " << newMaxBound << "\n";
                    std::cout << "Updated greedySet: ";
                    for (int i : greedySet) {
                        std::cout << i << " ";
                    }
                    std::cout << "\n";
                }
            }
        } else {
            maxHeap.push(std::pair<float, ll>(newMaxBound, currentMax.second));
        }
    }
		return greedyVector;
}
