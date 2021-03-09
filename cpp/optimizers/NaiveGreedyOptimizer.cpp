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

std::vector<std::pair<ll, float>> NaiveGreedyOptimizer::maximize(SetFunction &f_obj, float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbosity=false) {
	//TODO: take care of handling equal guys later
	//TODO: take care of different sizes of each items - becomes a candidate only if best and within budget, cost sensitive selection
	std::vector<std::pair<ll, float>>greedyVector;
	std::unordered_set<ll> greedySet;
	float rem_budget = budget;
	std::unordered_set<ll> groundSet = f_obj.getEffectiveGroundSet();

	if (verbosity) {
		std::cout << "Ground set:" << std::endl;
		for(int i: groundSet) {
			std::cout << i << " ";
		}
		std::cout << "\n";
		std::cout << "Num elements in groundset = " << groundSet.size() << std::endl;
		std::cout<<"Starting the naive greedy algorithm\n";
		std::cout << "Initial greedy set:" << std::endl;
		for(int i: greedySet) {
			std::cout << i << " ";
		}
		std::cout << "\n";
	}
	f_obj.clearMemoization();
	ll best_id;
	float best_val;
	while (rem_budget > 0) {
		best_id = -1;
		best_val = -1 * std::numeric_limits<float>::max();
		for (auto it = groundSet.begin(); it != groundSet.end(); ++it) {
			ll i = *it;
			if (greedySet.find(i) != greedySet.end()) { 
				//if this datapoint has already been included in greedySet, skip it
				continue;
			}
			float gain = f_obj.marginalGainWithMemoization(greedySet, i);
			//std::cout << "Gain of " << i << " is " << gain << "\n";
			if (gain > best_val) {
				best_id = i;
				best_val = gain;
			}
		}
		if(verbosity) {
            if(best_id == -1) throw "Nobody had greater gain than minus infinity!!";
			std::cout << "Next best item to add is " << best_id << " and its value addition is " << best_val << "\n";
        }
		if ( (best_val < 0 && stopIfNegativeGain) || (equals(best_val, 0, 1e-5) && stopIfZeroGain) ) {
			break;
		} else {
			f_obj.updateMemoization(greedySet, best_id);
			greedySet.insert(best_id); //greedily insert the best datapoint index of current iteration of while loop
			greedyVector.push_back(std::pair<ll, float>(best_id, best_val));
			rem_budget-=1;
			if(verbosity) {
				std::cout<<"Added element "<< best_id << " and the gain is " << best_val <<"\n";
				std::cout << "Updated greedySet: ";
		        for(int i: greedySet) {
			        std::cout << i << " ";
		        } 
		        std::cout << "\n";
			}
		}
	}
	return greedyVector;
}

