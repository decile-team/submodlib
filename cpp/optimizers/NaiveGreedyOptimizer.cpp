#include<iostream>
#include<set>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<utility>
#include"NaiveGreedyOptimizer.h"

NaiveGreedyOptimizer::NaiveGreedyOptimizer(){}

bool NaiveGreedyOptimizer::equals(double val1, double val2, double eps) {
  if (abs(val1 - val2) < eps)
    return true;
  else {
    return false;
  }
}

std::vector<std::pair<ll, double>> NaiveGreedyOptimizer::maximize(SetFunction &f_obj, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbose, bool showProgress, const std::vector<float>& costs, bool costSensitiveGreedy) {
	//TODO: take care of handling equal guys later
	std::vector<std::pair<ll, double>>greedyVector;
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
		for(ll i: groundSet) {
			std::cout << i << " ";
		}
		std::cout << "\n";
		std::cout << "Num elements in groundset = " << groundSet.size() << std::endl;
		std::cout << "Costs:" << std::endl;
		for(float i: costs) {
			std::cout << i << " ";
		}
		std::cout << "\n";
		std::cout << "Cost sensitive greedy: " << costSensitiveGreedy << "\n";
		std::cout<<"Starting the naive greedy algorithm\n";
		std::cout << "Initial greedy set:" << std::endl;
		for(int i: greedySet) {
			std::cout << i << " ";
		}
		std::cout << "\n";
	}
	f_obj.clearMemoization();
	ll best_id;
	double best_val;
	int step = 1;
	int displayNext = step;
	int percent = 0;
	float N = rem_budget;
	int iter = 0;
	if(costs.size()==0 && !costSensitiveGreedy) {
			while (rem_budget > 0) {
				best_id = -1;
				best_val = -1 * std::numeric_limits<double>::max();
				//for (auto it = groundSet.begin(); it != groundSet.end(); ++it) {
				for (auto i: groundSet) {
					//ll i = *it;
					if (greedySet.find(i) != greedySet.end()) { 
						//if this datapoint has already been included in greedySet, skip it
						continue;
					}
					double gain = f_obj.marginalGainWithMemoization(greedySet, i, false);
					if(verbose) std::cout << "Gain of " << i << " is " << gain << "\n";
					if (gain > best_val) {
						best_id = i;
						best_val = gain;
					}
				}
				if(verbose) {
					if(best_id == -1) throw "Nobody had greater gain than minus infinity!!";
					std::cout << "Next best item to add is " << best_id << " and its value addition is " << best_val << "\n";
				}
				if ( (best_val < 0 && stopIfNegativeGain) || (equals(best_val, 0, 1e-5) && stopIfZeroGain) ) {
					break;
				} else {
					f_obj.updateMemoization(greedySet, best_id);
					greedySet.insert(best_id); //greedily insert the best datapoint index of current iteration of while loop
					greedyVector.push_back(std::pair<ll, double>(best_id, best_val));
					rem_budget-=1;
					if(verbose) {
						std::cout<<"Added element "<< best_id << " and the gain is " << best_val <<"\n";
						std::cout << "Updated greedySet: ";
								for(int i: greedySet) {
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
		}
	} else if(costs.size()!= 0 && !costSensitiveGreedy) {
		bool possible = true;
		while (possible && rem_budget > 0) {
				best_id = -1;
				best_val = -1 * std::numeric_limits<double>::max();
				bool atLeastOneFound = false;
				//for (auto it = groundSet.begin(); it != groundSet.end(); ++it) {
				for (auto i: groundSet) {
					//ll i = *it;
					if (costs[i] > rem_budget) {
						//if this item can't fit in the remaining budget, skip it
						continue;
					}
					if (greedySet.find(i) != greedySet.end()) { 
						//if this datapoint has already been included in greedySet, skip it
						continue;
					}
					atLeastOneFound = true;
					double gain = f_obj.marginalGainWithMemoization(greedySet, i, false);
					if(verbose) std::cout << "Gain of " << i << " is " << gain << "\n";
					if (gain > best_val) {
						best_id = i;
						best_val = gain;
					}
				}
				if(!atLeastOneFound) {
					possible = false;
					continue;
				}
				if(verbose) {
					if(best_id == -1) throw "Nobody had greater gain than minus infinity!!";
					std::cout << "Next best item to add is " << best_id << " and its value addition is " << best_val << "\n";
				}
				if ( (best_val < 0 && stopIfNegativeGain) || (equals(best_val, 0, 1e-5) && stopIfZeroGain) ) {
					break;
				} else {
					f_obj.updateMemoization(greedySet, best_id);
					greedySet.insert(best_id); //greedily insert the best datapoint index of current iteration of while loop
					greedyVector.push_back(std::pair<ll, double>(best_id, best_val));
					rem_budget-=costs[best_id];
					if(verbose) {
						std::cout<<"Added element "<< best_id << " and the gain is " << best_val <<"\n";
						std::cout << "Remaining budget = " << rem_budget << "\n";
						std::cout << "Updated greedySet: ";
								for(int i: greedySet) {
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
		}
	} else if(costs.size()!= 0 && costSensitiveGreedy) {
		bool possible = true;
		while (possible && rem_budget > 0) {
				best_id = -1;
				best_val = -1 * std::numeric_limits<double>::max();
				//for (auto it = groundSet.begin(); it != groundSet.end(); ++it) {
				bool atLeastOneFound = false;
				for (auto i: groundSet) {
					//ll i = *it;
					if (costs[i] > rem_budget) {
						//if this item can't fit in the remaining budget, skip it
						continue;
					}
					if (greedySet.find(i) != greedySet.end()) { 
						//if this datapoint has already been included in greedySet, skip it
						continue;
					}
					atLeastOneFound = true;
					double gainPerCost = (f_obj.marginalGainWithMemoization(greedySet, i, false))/costs[i];
					if(verbose) std::cout << "Gain per cost of " << i << " is " << gainPerCost << "\n";
					if (gainPerCost > best_val) {
						best_id = i;
						best_val = gainPerCost;
					}
				}
				if(!atLeastOneFound) {
					possible = false;
					continue;
				}
				if(verbose) {
					if(best_id == -1) throw "Nobody had greater gain per cost than minus infinity!!";
					std::cout << "Next best item to add is " << best_id << " and its value addition is " << best_val << "\n";
				}
				if ( (best_val < 0 && stopIfNegativeGain) || (equals(best_val, 0, 1e-5) && stopIfZeroGain) ) {
					break;
				} else {
					f_obj.updateMemoization(greedySet, best_id);
					greedySet.insert(best_id); //greedily insert the best datapoint index of current iteration of while loop
					greedyVector.push_back(std::pair<ll, double>(best_id, best_val));
					rem_budget-=costs[best_id];
					if(verbose) {
						std::cout<<"Added element "<< best_id << " and the gain per cost is " << best_val <<"\n";
						std::cout << "Remaining budget = " << rem_budget << "\n";
						std::cout << "Updated greedySet: ";
								for(int i: greedySet) {
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
		}
	} else {
		throw "Must specify costs for costSensitiveGreedy variant";
	}
	return greedyVector;
}