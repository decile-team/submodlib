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

//TODO: perf: try emplace_back instead of push_back()

std::vector<std::pair<ll, double>> LazyGreedyOptimizer::maximize(
    SetFunction &f_obj, float budget, bool stopIfZeroGain,
    bool stopIfNegativeGain, bool verbose, bool showProgress, const std::vector<float>& costs, bool costSensitiveGreedy) {
    //TODO: take care of handling equal guys later
    std::vector<std::pair<ll, double>> greedyVector;
    std::unordered_set<ll> greedySet;
    if(costs.size()==0) {
		//every element is of same size, budget corresponds to cardinality
        greedyVector.reserve(budget);
	    greedySet.reserve(budget);
	}
    float rem_budget = budget;
    std::unordered_set<ll> groundSet = f_obj.getEffectiveGroundSet();
    if (verbose) {
        std::cout << "Ground set:" << std::endl;
        for (ll i : groundSet) {
            std::cout << i << " ";
        }
        std::cout << "\n";
        std::cout << "Num elements in groundset = " << groundSet.size()
                  << std::endl;
        std::cout << "Costs:" << std::endl;
		for(float i: costs) {
			std::cout << i << " ";
		}
		std::cout << "\n";
		std::cout << "Cost sensitive greedy: " << costSensitiveGreedy << "\n";
        std::cout << "Starting the lazy greedy algorithm\n";
        std::cout << "Initial greedy set:" << std::endl;
        for (int i : greedySet) {
            std::cout << i << " ";
        }
        std::cout << "\n";
    }
    f_obj.clearMemoization();
    //TODO:performance options:
    //1. default
    //2. vector with reserved space
    //3. deque
    // initialize priority queue:
    //reserve space for fast performance
    std::vector<std::pair<double, ll>> container;
    container.reserve(groundSet.size());
    std::priority_queue<std::pair<double, ll>, std::vector<std::pair<double, ll>>, std::less<std::pair<double, ll>>> maxHeap(std::less<std::pair<double, ll>>(), move(container));
    //std::priority_queue<std::pair<double, ll>> maxHeap;
    if(costSensitiveGreedy) {
        // for each element in the ground set
        for (auto elem : groundSet) {
            // store <elem, marginalGainWithMemoization(greedySet, elem)> in
            // priority-queue (max-heap)
            maxHeap.push(std::pair<double, ll>(
                (f_obj.marginalGainWithMemoization(greedySet, elem, false))/costs[elem], elem));
        }
    } else {
        // for each element in the ground set
        for (auto elem : groundSet) {
            // store <elem, marginalGainWithMemoization(greedySet, elem)> in
            // priority-queue (max-heap)
            maxHeap.push(std::pair<double, ll>(
                f_obj.marginalGainWithMemoization(greedySet, elem, false), elem));
        }
    }
    
    if(verbose) std::cout << "Max heap constructed\n";
    int step = 1;
	int displayNext = step;
	int percent = 0;
    float N = rem_budget;
    int iter = 0;
    if(costs.size()==0 && !costSensitiveGreedy) {
        while (rem_budget > 0) {
            std::pair<double, ll> currentMax = maxHeap.top();
            maxHeap.pop();
            if(verbose) std::cout << "currentMax element: " << currentMax.second <<" and its uper bound: " << currentMax.first << "\n";
            double newMaxBound =
                f_obj.marginalGainWithMemoization(greedySet, currentMax.second, false);
            if(verbose) {
                std::cout << "newMaxBound: " << newMaxBound <<"\n";
                std::cout << "nextBest element: " << maxHeap.top().second <<"and its bound: " << maxHeap.top().first << "\n";
            }
            if (newMaxBound >= maxHeap.top().first) {
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
                        std::pair<ll, double>(currentMax.second, newMaxBound));
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
                    if(showProgress) {
                        //TODO: use py::print
                        percent = (int)(((iter+1.0)/N)*100);
                        if (percent >= displayNext) {
                            //cout << "\r" << "[" << std::string(percent / 5, (char)254u) << std::string(100 / 5 - percent / 5, ' ') << "]";
                            std::cerr << "\r" << "[" << std::string(percent / 5, '|') << std::string(100 / 5 - percent / 5, ' ') << "]";
                            std::cerr << percent << "%" << " [Iteration " << iter + 1 << " of " << N << "]";
                            std::cerr.flush();
                            displayNext += step;
                        }
                        iter += 1;
                    }
                }
            } else {
                maxHeap.push(std::pair<double, ll>(newMaxBound, currentMax.second));
            }
        }
    } else if(costs.size()!= 0 && !costSensitiveGreedy) {
        while (rem_budget > 0 && maxHeap.size()>0) {
            std::pair<double, ll> currentMax = maxHeap.top();
            maxHeap.pop();
            if(verbose) std::cout << "currentMax element: " << currentMax.second <<" and its uper bound: " << currentMax.first << "\n";
            if(costs[currentMax.second] > rem_budget) {
                //can't add this guy to the greedy set as remaining budget is not sufficient
                //discard it
                continue;
            }
            double newMaxBound =
                f_obj.marginalGainWithMemoization(greedySet, currentMax.second, false);
            if(verbose) {
                std::cout << "newMaxBound: " << newMaxBound <<"\n";
                std::cout << "nextBest element: " << maxHeap.top().second <<"and its bound: " << maxHeap.top().first << "\n";
            }
            if (newMaxBound >= maxHeap.top().first) {
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
                        std::pair<ll, double>(currentMax.second, newMaxBound));
                    rem_budget -= costs[currentMax.second];
                    if (verbose) {
                        std::cout << "Added element " << currentMax.second
                                << " and the gain is " << newMaxBound << "\n";
                        std::cout << "Updated greedySet: ";
                        for (int i : greedySet) {
                            std::cout << i << " ";
                        }
                        std::cout << "\n";
                    }
                    if(showProgress) {
                        //TODO: use py::print
                        percent = (int)(((iter+1.0)/N)*100);
                        if (percent >= displayNext) {
                            //cout << "\r" << "[" << std::string(percent / 5, (char)254u) << std::string(100 / 5 - percent / 5, ' ') << "]";
                            std::cerr << "\r" << "[" << std::string(percent / 5, '|') << std::string(100 / 5 - percent / 5, ' ') << "]";
                            std::cerr << percent << "%" << " [Iteration " << iter + 1 << " of " << N << "]";
                            std::cerr.flush();
                            displayNext += step;
                        }
                        iter += 1;
                    }
                }
            } else {
                maxHeap.push(std::pair<double, ll>(newMaxBound, currentMax.second));
            }
        }
    } else if(costs.size()!= 0 && costSensitiveGreedy) {
        while (rem_budget > 0 && maxHeap.size()>0) {
            std::pair<double, ll> currentMax = maxHeap.top();
            maxHeap.pop();
            if(verbose) std::cout << "currentMax element: " << currentMax.second <<" and its uper bound: " << currentMax.first << "\n";
            if(costs[currentMax.second] > rem_budget) {
                //can't add this guy to the greedy set as remaining budget is not sufficient
                //discard it
                continue;
            }
            double newMaxBound =
                (f_obj.marginalGainWithMemoization(greedySet, currentMax.second, false))/costs[currentMax.second];
            if(verbose) {
                std::cout << "newMaxBound: " << newMaxBound <<"\n";
                std::cout << "nextBest element: " << maxHeap.top().second <<"and its bound: " << maxHeap.top().first << "\n";
            }
            if (newMaxBound >= maxHeap.top().first) {
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
                        std::pair<ll, double>(currentMax.second, newMaxBound));
                    rem_budget -= costs[currentMax.second];
                    if (verbose) {
                        std::cout << "Added element " << currentMax.second
                                << " and the gain is " << newMaxBound << "\n";
                        std::cout << "Updated greedySet: ";
                        for (int i : greedySet) {
                            std::cout << i << " ";
                        }
                        std::cout << "\n";
                    }
                    if(showProgress) {
                        //TODO: use py::print
                        percent = (int)(((iter+1.0)/N)*100);
                        if (percent >= displayNext) {
                            //cout << "\r" << "[" << std::string(percent / 5, (char)254u) << std::string(100 / 5 - percent / 5, ' ') << "]";
                            std::cerr << "\r" << "[" << std::string(percent / 5, '|') << std::string(100 / 5 - percent / 5, ' ') << "]";
                            std::cerr << percent << "%" << " [Iteration " << iter + 1 << " of " << N << "]";
                            std::cerr.flush();
                            displayNext += step;
                        }
                        iter += 1;
                    }
                }
            } else {
                maxHeap.push(std::pair<double, ll>(newMaxBound, currentMax.second));
            }
        }
    } else {
        throw "Must specify costs for costSensitiveGreedy variant";
    }
    
	return greedyVector;
}
