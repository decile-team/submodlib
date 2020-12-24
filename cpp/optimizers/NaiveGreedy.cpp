//Note that this code is in prototype state. Its a direct implementation of the pseudo code and there seems to be some logical errors.

#include<iostream>
#include<set>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include"NaiveGreedy.h"


std::vector<float> naiveGreedyMax(FacilityLocation f_obj, float budget)//Possibly we need a template type instead of a fixed type for f_obj 
{
	std::vector<float>greedyVector;
	std::set<ll>greedySet;
	float rem_budget = budget;
	
	while (rem_budget > 0)
	{
		//In pseudo code, a single variable "best" was used for storing both marginal gain and datapoint index which seemed incorrect.
		//Therefore, I have instead used 2 variables viz. best_i and best_val
		ll best_i;
		float best_val;
		for (auto it = f_obj.getEffectiveGroundSet().begin(); it != f_obj.getEffectiveGroundSet().end(); ++it)
		{
			ll i = *it;
			if (greedySet.find(i) != greedySet.end())
			{
				continue;
			}
			if (f_obj.marginalGainSequential(greedySet, i) > best_val)//Bug: uninitilized variable best_val is getting consumed
			{
				best_i = i;
				best_val = f_obj.marginalGainSequential(greedySet, i);
			}
		}

		//Bugs: Uninitilized variable best_i is getting used below.
		f_obj.sequentialUpdate(greedySet, best_i); 
		greedySet.insert(best_i); 
		greedyVector.push_back(best_i);//Verify weather greedyVector contains datapoint indicies or marginal gains
		--rem_budget;
	}

	return greedyVector;
}

