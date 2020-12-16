/*
Implementation decisions.
1) Considering the possibility of very large datasets, its safer to use long long int (alias ll) in place of int (for storing size/index of data)

2) Containers like X, groundset, effectiveGroundSet etc (which contain index of datapoints) have been implemented as set (instead of vector).
This is because in C++, set container is implemented as red-black tree and thus search operations happen in log(n) time which is beneficial
for functions like marginalGain(), sequentialUpdate() etc that require such search operations frequently.
If we use vectors then for efficiency we would have an additional responsibility of ensuring that they are sorted. Thus,
set is a more natural choice here

3) For sparse mode, constructor will accept sparse matrix as a collection of 3 component vectors (for csr) and use them to instantiate
a sparse matrix object either using a custom utility class or using some high performance library like boost.

Possible Improvement.
If order of elements in X, groundset, effectiveGroundSet etc doesn't matter, we can improve the performance further by using unordered_set instead of 
set. unordered_set is implemented using hashmap and has an average search complexity of O(1). Although, if this is done then we also need to write
a custom function for "Intersection" because set_intersection() only work on containers with sorted data. Also, to avoid excessive rehashing, we will
have to reserve a certain number of buckets in advance.

*/

#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"FacilityLocation.h"

typedef long long int ll;

//Note to self: Migrate all parameter related sanity/error checks from C++ FL to Python FL

//For dense mode
FacilityLocation::FacilityLocation(ll n_, std::string mode_, std::vector<std::vector<float>>k_dense_, ll num_neighbors_, bool partial_, std::set<ll> ground_)
{
	if (mode_ != "dense") 
	{
		std::cerr << "Error: Incorrect mode specified for the provided dense similarity matrix\n";
		return;
	}

	if (k_dense_.size() == 0)
	{
		std::cerr << "Error: Empty similarity matrix\n";
		return;
	}

	n = n_;
	mode = mode_;
	k_dense = k_dense_;
	num_neighbors = num_neighbors_;
	partial = partial_;
	if (partial == true)
	{
		effectiveGroundSet = ground_;
	}
	else
	{
		for (ll i = 0; i < n; ++i)
		{
			effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
		}
	}
	numEffectiveGroundset = effectiveGroundSet.size();
	similarityWithNearestInEffectiveX.resize(numEffectiveGroundset, 0);
}

//For sparse mode (TODO)
FacilityLocation::FacilityLocation(ll n_, std::string mode_, std::vector<float>arr_val, std::vector<float>arr_count, std::vector<float>arr_col, ll num_neighbors_, bool partial_, std::set<ll> ground_)
{
	std::cerr<<"To be implemented\n";
}

//For cluster mode
FacilityLocation::FacilityLocation(ll n_, std::string mode_, std::vector<std::set<ll>>clusters_, ll num_neighbors_, bool partial_, std::set<ll> ground_ )
{
	if (mode_ != "cluster")
	{
		std::cerr << "Error: Incorrect mode specified for the provided cluster\n";
		return;
	}

	if (clusters_.size() == 0)
	{
		std::cerr << "Error: Cluster vector is empty\n";
		return;
	}

	n = n_;
	mode = mode_;
	clusters = clusters_;
	num_neighbors = num_neighbors_;
	partial = partial_;
	if (partial == true)
	{
		effectiveGroundSet = ground_;
	}
	else
	{
		for (ll i = 0; i < n; ++i)
		{
			effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
		}
	}
	numEffectiveGroundset = effectiveGroundSet.size();
}

//helper friend function
float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj)
{
	ll i = datapoint_ind, j;
	auto it = dataset_ind.begin();
	float m = obj.k_dense[i][*it];
	
	for (; it != dataset_ind.end(); ++it)//search max similarity wrt datapoints of given dataset
	{
		ll j = *it;
		if (obj.k_dense[i][j] > m)
		{
			m = obj.k_dense[i][j];
		}
	}

	return m;
}


float FacilityLocation::evaluate(std::set<ll> X)
{
	std::set<ll> effectiveX;
	float result=0;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (mode == "dense")
	{
		//Implementing f(X)=Sum_i_V ( max_j_X ( s_ij ) )
		for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it) //O(n^2) where n=num of elements in effective GS 
		{
			ll ind = *it;
			result += get_max_sim_dense(ind, effectiveX, *this);
		}
	}
	else
	{
		if (mode == "sparse")
		{
			//TODO

		}
		else
		{
			if (mode == "clustered")
			{
				//TODO
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return result;
}


float FacilityLocation::evaluateSequential(std::set<ll> X) //assumes that pre computed statistics exist for effectiveX
{
	std::set<ll> effectiveX;
	float result = 0;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (mode == "dense")
	{
		for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it)
		{
			ll ind = *it;
			result += similarityWithNearestInEffectiveX[ind];
		}
	}
	else
	{
		if (mode == "sparse")
		{
			//TODO

		}
		else
		{
			if (mode == "clustered")
			{
				//TODO
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return result;
}


float FacilityLocation::marginalGain(std::set<ll> X, ll item)
{
	std::set<ll> effectiveX;
	float gain = 0;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (effectiveGroundSet.find(item) == effectiveGroundSet.end()) //O(log(n))
	{
		return 0;
	}

	if (effectiveX.find(item) != effectiveX.end())
	{
		return 0;
	}

	if (mode == "dense")
	{
		for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it)
		{
			ll ind = *it;
			float m = get_max_sim_dense(ind, effectiveX, *this);
			if (k_dense[ind][item] > m)
			{
				gain += (k_dense[ind][item] - m);
			}
		}
	}
	else
	{
		if (mode == "sparse")
		{
			//TODO

		}
		else
		{
			if (mode == "clustered")
			{
				//TODO
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return gain;
}


float FacilityLocation::marginalGainSequential(std::set<ll> X, ll item)
{
	std::set<ll> effectiveX;
	float gain = 0;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (effectiveGroundSet.find(item) == effectiveGroundSet.end())
	{
		return 0;
	}

	if (effectiveX.find(item) != effectiveX.end())
	{
		return 0;
	}

	if (mode == "dense")
	{
		for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it)
		{
			ll ind = *it;
			if (k_dense[ind][item] > similarityWithNearestInEffectiveX[ind])
			{
				gain += (k_dense[ind][item] - similarityWithNearestInEffectiveX[ind]);
			}

		}
	}
	else
	{
		if (mode == "sparse")
		{
			//TODO

		}
		else
		{
			if (mode == "clustered")
			{
				//TODO
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return gain;
}

void FacilityLocation::sequentialUpdate(std::set<ll> X, ll item)
{
	std::set<ll> effectiveX;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (mode == "dense")
	{
		if (effectiveGroundSet.find(item) == effectiveGroundSet.end())
		{
			return;
		}

		for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it)
		{
			ll ind = *it;
			if (k_dense[ind][item] > similarityWithNearestInEffectiveX[ind])
			{
				similarityWithNearestInEffectiveX[ind] = k_dense[ind][item];
			}

		}
	}
	else
	{
		if (mode == "sparse")
		{
			//TODO

		}
		else
		{
			if (mode == "clustered")
			{
				//TODO
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}
}

std::set<ll> FacilityLocation::getEffectiveGroundSet()
{
	return effectiveGroundSet;
}


