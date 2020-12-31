//Note that this code is in prototype state. Its a direct implementation of the pseudo code and there seems to be some logical errors.

#include<iostream>
#include<set>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<utility>
#include"NaiveGreedyOptimizer.h"

NaiveGreedyOptimizer::NaiveGreedyOptimizer(){}
//NaiveGreedyOptimizer::NaiveGreedyOptimizer(FacilityLocation obj_):f_obj(obj_) {}

std::vector<std::pair<ll, float>> NaiveGreedyOptimizer::maximize(SetFunction &f_obj, float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbosity=false)
//std::vector<std::pair<int, float>> NaiveGreedyOptimizer::maximize(float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbosity=false)
{
	std::vector<std::pair<ll, float>>greedyVector;
	std::set<ll>greedySet;
	float rem_budget = budget;
	
	while (rem_budget > 0)
	{
		ll best_i = -1;
		float best_val = -1 * std::numeric_limits<float>::max();
		for (auto it = f_obj.getEffectiveGroundSet().begin(); it != f_obj.getEffectiveGroundSet().end(); ++it)
		{
			ll i = *it;
			if (greedySet.find(i) != greedySet.end())//if a datapoint has already been included in greedySet, skip to next datapoint
			{
				continue;
			}
			float gain = f_obj.marginalGainSequential(greedySet, i);
			if (gain > best_val)
			{
				best_i = i;
				best_val = gain;
			}
		}
		if ( (best_val < 0 && stopIfNegativeGain) || (best_val == 0 && stopIfZeroGain) ) 
		{
			break;
		}
		else
		{
			f_obj.sequentialUpdate(greedySet, best_i); //memoize the result of current iteration of while loop
			greedySet.insert(best_i); //greedily insert the best datapoint index of current iteration of while loop
			greedyVector.push_back(std::pair<ll, float>(best_i, best_val));
			rem_budget-=1;
		}
		
	}

	return greedyVector;
}

