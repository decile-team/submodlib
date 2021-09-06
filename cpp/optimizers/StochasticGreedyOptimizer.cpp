#include<iostream>
#include<set>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<utility>
#include<random>
#include"StochasticGreedyOptimizer.h"

StochasticGreedyOptimizer::StochasticGreedyOptimizer(){}

bool StochasticGreedyOptimizer::equals(double val1, double val2, double eps) {
  if (abs(val1 - val2) < eps)
    return true;
  else {
    return false;
  }
}

std::vector<std::pair<ll, double>> StochasticGreedyOptimizer::maximize(SetFunction &f_obj, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress, const std::vector<float>& costs, bool costSensitiveGreedy) {
	//TODO: take care of handling equal guys later
	//TODO: take care of different sizes of each items - becomes a candidate only if best and within budget, cost sensitive selection
	std::vector<std::pair<ll, double>> greedyVector;
	std::unordered_set<ll> greedySet;
	if(costs.size()==0) {
		//every element is of same size, budget corresponds to cardinality
		greedyVector.reserve(budget);
		greedySet.reserve(budget);
	}
	float rem_budget = budget;
	std::unordered_set<ll> remainingSet = f_obj.getEffectiveGroundSet();
	ll n = remainingSet.size();
	ll randomSetSize = ((double)n/budget)* log(1/epsilon);

	if (verbose) {
		std::cout << "Epsilon = " << epsilon << "\n";
		std::cout << "Random set size = " << randomSetSize << "\n";
		std::cout << "Ground set:" << std::endl;
		for(int i: remainingSet) {
			std::cout << i << " ";
		}
		std::cout << "\n";
		std::cout << "Num elements in groundset = " << remainingSet.size() << std::endl;
		std::cout<<"Starting the stochastic greedy algorithm\n";
		std::cout << "Initial greedy set:" << std::endl;
		for(int i: greedySet) {
			std::cout << i << " ";
		}
		std::cout << "\n";
	}
	f_obj.clearMemoization();
	srand(1);
	ll best_id;
	double best_val;
	int i = 0;
	int step = 1;
	int displayNext = step;
	int percent = 0;
  float N = rem_budget;
	int iter = 0;
	while (rem_budget > 0) {
		std::unordered_set<ll> randomSet;
		while(randomSet.size() < randomSetSize) {
		    ll elem = rand() % n; //TODO:serious issue, works only till RAND_MAX
				//std::cout << "Trying random element " << elem << "\n";
        if((remainingSet.find(elem) != remainingSet.end()) && (randomSet.find(elem) == randomSet.end())){
            //valid element
						//std::cout << "Valid, adding it\n";
						randomSet.insert(elem);
				}
		}
		if (verbose) {
            std::cout << "Iteration " << i << "\n";
            std::cout << "Random set = [";
            for (auto elem : randomSet) {
                std::cout << elem << " ";
            }
            std::cout << "\n";
						std::cout << "Now running naive greedy on the random set\n";
        }
		best_id = -1;
		best_val = -1 * std::numeric_limits<double>::max();
		//for (auto it = groundSet.begin(); it != groundSet.end(); ++it) {
		for (auto i: randomSet) {
			//ll i = *it;
			double gain = f_obj.marginalGainWithMemoization(greedySet, i, false);
			//std::cout << "Gain of " << i << " is " << gain << "\n";
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
			remainingSet.erase(best_id);
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
		i += 1;
	}
	return greedyVector;
}

