/*
Implementation decisions.
1) Considering the possibility of very large datasets, its safer to use long long int (alias ll) in place of int (for storing size/index of data)

2) Containers like X, groundset, effectiveGroundSet etc (which contain index of datapoints) have been implemented as set (instead of vector).
This is because in C++, set container is implemented as red-black tree and thus search operations happen in log(n) time which is beneficial
for functions like marginalGain(), updateMemoization() etc that require such search operations frequently.
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
#include "../utils/helper.h"
#include"FacilityLocation.h"

FacilityLocation::FacilityLocation(){}

//Constructor for dense mode
FacilityLocation::FacilityLocation(ll n_, std::vector<std::vector<float>>denseKernel_, bool partial_, std::unordered_set<ll> ground_, bool separateMaster_) {
	// std::cout << "FacilityLocation Dense Constructor\n";
	n = n_;
	mode = "dense";
	denseKernel = denseKernel_;
	partial = partial_;
	separateMaster = separateMaster_;
	if (partial == true) {
		//ground set will now be the subset provided
		effectiveGroundSet = ground_;
	}
	else {
		//create groundSet with items 0 to n-1
		for (ll i = 0; i < n; ++i){
			effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
		}
	}
	numEffectiveGroundset = effectiveGroundSet.size();
	
	if(separateMaster==true) {
		//populate a different master set
		n_master = denseKernel.size();	
		for (ll i = 0; i < n_master; ++i) {
			masterSet.insert(i); //each insert takes O(log(n)) time
		}
	}
	else {
		//master set will now be same as the ground set
		n_master=numEffectiveGroundset;
		masterSet=effectiveGroundSet;
	}
	similarityWithNearestInEffectiveX.resize(n_master, 0);
	if(partial == true) {
		ll ind = 0;
		for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it) {
			originalToPartialIndexMap[*it] = ind;
			ind += 1;
		}
	}
}

//For sparse mode
FacilityLocation::FacilityLocation(ll n_, std::vector<float>arr_val, std::vector<ll>arr_count, std::vector<ll>arr_col) {
	// std::cout << "FacilityLocation Sparse Constructor\n";
	if (arr_val.size() == 0 || arr_count.size() == 0 || arr_col.size() == 0) {
		throw "Error: Empty/Corrupt sparse similarity kernel";
	}
	n = n_;
	mode = "sparse";
	sparseKernel = SparseSim(arr_val, arr_count, arr_col);
	partial = false;
	separateMaster = false;
	//create groundSet with items 0 to nv-1
	for (ll i = 0; i < n; ++i) {
		effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();
	
	n_master=numEffectiveGroundset;
	masterSet=effectiveGroundSet;
	
	similarityWithNearestInEffectiveX.resize(n_master, 0);
}

//For cluster mode
FacilityLocation::FacilityLocation(ll n_, std::vector<std::unordered_set<ll>>clusters_,std::vector<std::vector<std::vector<float>>>clusterKernels_, std::vector<ll>clusterIndexMap_) {
	// std::cout << "FacilityLocation Clustered Constructor\n";
	n = n_;
	mode = "clustered";
	num_clusters = clusters_.size();
	clusters = clusters_;
	clusterKernels = clusterKernels_;
	clusterIndexMap = clusterIndexMap_;
	partial = false;
	separateMaster = false;

	//create groundSet with items 0 to nv-1
	for (ll i = 0; i < n; ++i) {
		effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();

	n_master=numEffectiveGroundset;
	masterSet=effectiveGroundSet;

	clusterIDs.resize(n);
	for(int i=0;i<num_clusters;++i) {  //O(n) (One time operation)
		std::unordered_set<ll> ci=clusters[i];
		for (auto it = ci.begin(); it != ci.end(); ++it) {
			ll ind = *it;
			clusterIDs[ind]=i;
		}
	}

	relevantX.resize(num_clusters);
	clusteredSimilarityWithNearestInRelevantX.resize(n, 0);
	
	// for(ll i=0;i<num_clusters;++i) {
	// 	settemp;
	// 	relevantX.push_back(temp);
	// }
}

//helper friend function
float get_max_sim_dense(ll datapoint_ind, std::unordered_set<ll> dataset_ind, FacilityLocation obj) {
	if(dataset_ind.size()==0) {
		return 0;
	}
	ll i = datapoint_ind, j; //i comes from masterSet and j comes from X (which is a subset of groundSet)
	auto it = dataset_ind.begin();
	float m = obj.denseKernel[i][*it];

	for (; it != dataset_ind.end(); ++it) {//search max similarity wrt datapoints of given dataset
		ll j = *it;
		if (obj.denseKernel[i][j] > m) {
			m = obj.denseKernel[i][j];
		}
	}

	return m;
}
float get_max_sim_sparse(ll datapoint_ind, std::unordered_set<ll> dataset_ind, FacilityLocation obj) {
	if(dataset_ind.size()==0) {
		return 0;
	}
	ll i = datapoint_ind, j; //i comes from masterSet and j comes from X (which is a subset of groundSet)
	auto it = dataset_ind.begin();
	float m = obj.sparseKernel.get_val(i,*it);
	
	for (; it != dataset_ind.end(); ++it) {//search max similarity wrt datapoints of given dataset
		ll j = *it;
		float temp_val = obj.sparseKernel.get_val(i,j);
		if (temp_val > m) {
			m = temp_val;
		}
	}

	return m;
}

float get_max_sim_cluster(ll datapoint_ind, std::unordered_set<ll> dataset_ind, FacilityLocation obj, ll cluster_id) {
    if(dataset_ind.size()==0) {
		return 0;
	}
	ll i = datapoint_ind, j, i_, j_; 
	auto it = dataset_ind.begin();
	i_ = obj.clusterIndexMap[i];
	j_ = obj.clusterIndexMap[*it]; 
	float m = obj.clusterKernels[cluster_id][i_][j_];
	//Possibly transform i,j to local kernel index

	for (; it != dataset_ind.end(); ++it) {
		ll j = *it;
		//Obtain local kernel indicies for given cluster
		i_ = obj.clusterIndexMap[i];
		j_ = obj.clusterIndexMap[j];
		if (obj.clusterKernels[cluster_id][i_][j_] > m) {
			m = obj.clusterKernels[cluster_id][i_][j_];
		}
	}
	return m;
}

float FacilityLocation::evaluate(std::unordered_set<ll> X) {
	// std::cout << "FacilityLocation evaluate\n";
	std::unordered_set<ll> effectiveX;
	float result=0;

	if (partial == true) {
		//effectiveX = intersect(X, effectiveGroundSet)
		// std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
		effectiveX = set_intersection(X, effectiveGroundSet);
	} else {
		effectiveX = X;
	}

	if(effectiveX.size()==0) {
		return 0;
	}

	if (mode == "dense") {
		//for each element in master set
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) { //O(n^2) where n=num of elements in effective GS 
			ll ind = *it;
			//find max similarity of i with all items in effectiveX
			result += get_max_sim_dense(ind, effectiveX, *this);
		}
	} else if (mode == "sparse") {
        for (auto it = masterSet.begin(); it != masterSet.end(); ++it) { //O(n^2) where n=num of elements in effective GS 
				ll ind = *it;
				result += get_max_sim_sparse(ind, effectiveX, *this);
		}
	} else {
		//for each cluster
		for(ll i=0;i<num_clusters;++i) {
			std::unordered_set<ll> releventSubset;
			std::unordered_set<ll> ci = clusters[i];
			// std::set_intersection(X.begin(), X.end(), ci.begin(), ci.end(), std::inserter(releventSubset, releventSubset.begin()));
			releventSubset = set_intersection(X, ci);

			if(releventSubset.size()==0) { //if no intersection, skip to next cluster
				continue;
			}
			
			for (auto it = ci.begin(); it != ci.end(); ++it) {
				ll ind = *it;
				result += get_max_sim_cluster(ind, releventSubset, *this, i);
			}
		}
	}
	return result;
}


float FacilityLocation::evaluateWithMemoization(std::unordered_set<ll> X) { 
	// std::cout << "FacilityLocation evaluateWithMemoization\n";
    //assumes that appropriate pre computed memoized statistics exist for effectiveX

	std::unordered_set<ll> effectiveX;
	float result = 0;

	if (partial == true) {
		//effectiveX = intersect(X, effectiveGroundSet)
		// std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
		effectiveX = set_intersection(X, effectiveGroundSet);
	} else {
		effectiveX = X;
	}

	if(effectiveX.size()==0) {
		return 0;
	}

	if (mode == "dense" || mode == "sparse") {
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
			ll ind = *it;
			result += similarityWithNearestInEffectiveX[(partial)?originalToPartialIndexMap[ind]:ind];
		}
	} else {
		for(ll i=0;i<num_clusters;++i) {
			if(relevantX[i].size()==0) {
				continue;
			}
			std::unordered_set<ll> ci = clusters[i];
			for (auto it = ci.begin(); it != ci.end(); ++it) {
				ll ind = *it;
				result += clusteredSimilarityWithNearestInRelevantX[ind];
			}
		}
	}
	return result;
}


float FacilityLocation::marginalGain(std::unordered_set<ll> X, ll item) {
	// std::cout << "FacilityLocation marginalGain\n";
	std::unordered_set<ll> effectiveX;
	float gain = 0;

	if (partial == true) {
		//effectiveX = intersect(X, effectiveGroundSet)
		// std::(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
		effectiveX = set_intersection(X, effectiveGroundSet);
	} else {
		effectiveX = X;
	}

	if (effectiveX.find(item)!=effectiveX.end()) {
		return 0;
	}

	if (mode == "dense") {
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
			ll ind = *it;
			float m = get_max_sim_dense(ind, effectiveX, *this);
			if (denseKernel[ind][item] > m) {
				gain += (denseKernel[ind][item] - m);
			}
		}
	} else if (mode == "sparse") {
			for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
				ll ind = *it;
				float m = get_max_sim_sparse(ind, effectiveX, *this);
				float temp_val = sparseKernel.get_val(ind,item);
				if (temp_val > m) {
					gain += (temp_val - m);
				}
			}
	} else {
        ll i = clusterIDs[item];
		std::unordered_set<ll> releventSubset;
		std::unordered_set<ll> ci = clusters[i];
		// std::set_intersection(X.begin(), X.end(), ci.begin(), ci.end(), std::inserter(releventSubset, releventSubset.begin()));
		releventSubset = set_intersection(X, ci);
		if(releventSubset.size()==0) {
			for (auto it = ci.begin(); it != ci.end(); ++it) {
				ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				ll item_ = clusterIndexMap[item];
				gain+=clusterKernels[i][ind_][item_];
			}
		} else {
			for (auto it = ci.begin(); it != ci.end(); ++it) {
				ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				ll item_ = clusterIndexMap[item];
				float m = get_max_sim_cluster(ind, releventSubset, *this, i);
				if (clusterKernels[i][ind_][item_] > m) {
					gain += (clusterKernels[i][ind_][item_] - m);
				}
			}
		}
	}
	return gain;
}


float FacilityLocation::marginalGainWithMemoization(std::unordered_set<ll> X, ll item) {
	// std::cout << "FacilityLocation marginalGainWithMemoization\n";
	std::unordered_set<ll> effectiveX;
	float gain = 0;
	if (partial == true) {
		// std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	    effectiveX = set_intersection(X, effectiveGroundSet);
	} else {
		effectiveX = X;
	}
	if (effectiveX.find(item)!=effectiveX.end()) {
		return 0;
	}
	if (mode == "dense") {
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
			ll ind = *it;
			if (denseKernel[ind][item] > similarityWithNearestInEffectiveX[(partial)?originalToPartialIndexMap[ind]:ind]) {
				gain += (denseKernel[ind][item] - similarityWithNearestInEffectiveX[(partial)?originalToPartialIndexMap[ind]:ind]);
			}
		}
	} else if (mode == "sparse") {
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
			ll ind = *it;
			float temp_val = sparseKernel.get_val(ind,item);
			if (temp_val > similarityWithNearestInEffectiveX[ind]) {
				gain += (temp_val - similarityWithNearestInEffectiveX[ind]);
			}
		}
	} else {
        ll i = clusterIDs[item];
		std::unordered_set<ll> releventSubset = relevantX[i];
		std::unordered_set<ll> ci = clusters[i];
		
		if(releventSubset.size()==0) {
			for (auto it = ci.begin(); it != ci.end(); ++it) {
				ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				ll item_ = clusterIndexMap[item];
				gain+=clusterKernels[i][ind_][item_];
			}
		} else {
			for (auto it = ci.begin(); it != ci.end(); ++it) {
				ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				ll item_ = clusterIndexMap[item];
				float temp_val = clusterKernels[i][ind_][item_];
				if (temp_val > clusteredSimilarityWithNearestInRelevantX[ind]) {
					gain += (temp_val - clusteredSimilarityWithNearestInRelevantX[ind]);
				}
			}
		}
	}
	return gain;
}

void FacilityLocation::updateMemoization(std::unordered_set<ll> X, ll item) {
	// std::cout << "FacilityLocation updateMemoization\n";
	std::unordered_set<ll> effectiveX;

	if (partial == true) {
		//effectiveX = intersect(X, effectiveGroundSet)
		// std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
		effectiveX = set_intersection(X, effectiveGroundSet);
	} else {
		effectiveX = X;
	}
	if (effectiveX.find(item)!=effectiveX.end()) {
		return;
	}
	if (mode == "dense") {
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
			ll ind = *it;
			if (denseKernel[ind][item] > similarityWithNearestInEffectiveX[(partial)?originalToPartialIndexMap[ind]:ind]) {
				similarityWithNearestInEffectiveX[(partial)?originalToPartialIndexMap[ind]:ind] = denseKernel[ind][item];
			}
		}
	} else if (mode == "sparse") {
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
			ll ind = *it;
			float temp_val = sparseKernel.get_val(ind,item);
			if (temp_val > similarityWithNearestInEffectiveX[ind]) {
				similarityWithNearestInEffectiveX[ind] = temp_val;
			}
		}
	} else {
        ll i = clusterIDs[item];
		std::unordered_set<ll> ci = clusters[i];
		for (auto it = ci.begin(); it != ci.end(); ++it) {
			ll ind = *it;
			ll ind_=clusterIndexMap[ind];
			ll item_ = clusterIndexMap[item];
			float temp_val = clusterKernels[i][ind_][item_];	
			if (temp_val > clusteredSimilarityWithNearestInRelevantX[ind]) {
				clusteredSimilarityWithNearestInRelevantX[ind]= temp_val;
			}		 
		}
		relevantX[i].insert(item);
	}
}

std::unordered_set<ll> FacilityLocation::getEffectiveGroundSet() {
	// std::cout << "FacilityLocation getEffectiveGroundSet\n";
	return effectiveGroundSet;
}


std::vector<std::pair<ll, float>> FacilityLocation::maximize(std::string optimizer,float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbosity=false) {
	// std::cout << "FacilityLocation maximize\n";
	if(optimizer == "NaiveGreedy") {
		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbosity);
	} 
}


void FacilityLocation::cluster_init(ll n_, std::vector<std::vector<float>>denseKernel_, std::unordered_set<ll> ground_, bool partial) {
	// std::cout << "FacilityLocation clusterInit\n";
	*this = FacilityLocation(n_, denseKernel_, partial, ground_, false);
}

void FacilityLocation::clearMemoization() {
	// std::cout << "FacilityLocation clearMemoization\n";
	//We could do https://stackoverflow.com/questions/55266468/whats-the-fastest-way-to-reinitialize-a-vector/55266856 to replace it with a more efficient implementation. However, clear() and assign() involve re-alloc and hence are slower. fill() also involves a loop. memset could be faster, but being at lower level, could be unsafe to use

    //reset similarityWithNearestInEffectiveX for dense and sparse modes
	if(mode=="dense" || mode=="sparse") {
		for(int i=0;i<n_master;++i) {
			similarityWithNearestInEffectiveX[i]=0;
		}
	} else {
	    //reset relevantX and clusteredSimilarityWithNearestInRelevantX for clustered mode
		for(int i=0;i<num_clusters;++i) {
			relevantX[i].clear();
		}
		for(int i=0;i<n;++i) {
			clusteredSimilarityWithNearestInRelevantX[i]=0;
		}
	}
}

void FacilityLocation::setMemoization(std::unordered_set<ll> X) 
{
	// std::cout << "FacilityLocation setMemoization\n";
    clearMemoization();
    std::unordered_set<ll> temp;
	for (auto it = X.begin(); it != X.end(); ++it)
	{	
		updateMemoization(temp, *it);
		temp.insert(*it);	
	}
}


