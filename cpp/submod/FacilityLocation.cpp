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

//Constructor for dense mode (kenel supplied)
FacilityLocation::FacilityLocation(ll n_, std::vector<std::vector<float>> const &denseKernel_, bool partial_, std::unordered_set<ll> const &ground_, bool separateMaster_): n(n_), mode(dense), denseKernel(denseKernel_), partial(partial_), separateMaster(separateMaster_)  {
	// std::cout << "FacilityLocation Dense Constructor\n";
	//n = n_;
	//mode = dense;
	//denseKernel = denseKernel_;
	//partial = partial_;
	//separateMaster = separateMaster_;
	if (partial == true) {
		//ground set will now be the subset provided
		effectiveGroundSet = ground_;
	}
	else {
		//create groundSet with items 0 to n-1
		effectiveGroundSet.reserve(n);
		for (ll i = 0; i < n; ++i){
			effectiveGroundSet.insert(i); //each insert takes O(1) time
		}
	}
	numEffectiveGroundset = effectiveGroundSet.size();
	
	if(separateMaster==true) {
		//populate a different master set
		n_master = denseKernel.size();	
		masterSet.reserve(n_master);
		for (ll i = 0; i < n_master; ++i) {
			masterSet.insert(i); //each insert takes O(1) time
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
		//for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it) {
		for (auto it: effectiveGroundSet) {
			originalToPartialIndexMap[it] = ind;
			ind += 1;
		}
	}
}



//Constructor for dense mode (kenel not supplied)
FacilityLocation::FacilityLocation(ll n_, std::vector<std::vector<float>> &data, std::vector<std::vector<float>> &data_master, bool separateMaster_, std::string metric): n(n_), separateMaster(separateMaster_)  {
	if(separateMaster == true) {
		denseKernel = create_kernel_NS(data, data_master, metric);
	} else {
		denseKernel = create_square_kernel_dense(data, metric);
	}
	mode = dense;
	partial = false;

	//create groundSet with items 0 to n-1
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); //each insert takes O(1) time
	}
	
	numEffectiveGroundset = effectiveGroundSet.size();
	
	if(separateMaster==true) {
		//populate a different master set
		n_master = denseKernel.size();	
		masterSet.reserve(n_master);
		for (ll i = 0; i < n_master; ++i) {
			masterSet.insert(i); //each insert takes O(1) time
		}
	}
	else {
		//master set will now be same as the ground set
		n_master=numEffectiveGroundset;
		masterSet=effectiveGroundSet;
	}
	similarityWithNearestInEffectiveX.resize(n_master, 0);
}

//For sparse mode
FacilityLocation::FacilityLocation(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col): n(n_), mode(sparse), partial(false), separateMaster(false) {
	// std::cout << "FacilityLocation Sparse Constructor\n";
	if (arr_val.size() == 0 || arr_count.size() == 0 || arr_col.size() == 0) {
		throw "Error: Empty/Corrupt sparse similarity kernel";
	}
	//n = n_;
	//mode = sparse;
	sparseKernel = SparseSim(arr_val, arr_count, arr_col);
	//partial = false;
	//separateMaster = false;
	//create groundSet with items 0 to nv-1
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i) {
		effectiveGroundSet.insert(i); //each insert takes O(1) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();
	
	n_master=numEffectiveGroundset;
	masterSet=effectiveGroundSet;
	
	similarityWithNearestInEffectiveX.resize(n_master, 0);
}

//For cluster mode
FacilityLocation::FacilityLocation(ll n_, std::vector<std::unordered_set<ll>> const &clusters_,std::vector<std::vector<std::vector<float>>> const &clusterKernels_, std::vector<ll> const &clusterIndexMap_): n(n_), mode(clustered), num_clusters(clusters_.size()), clusters(clusters_), clusterKernels(clusterKernels_), clusterIndexMap(clusterIndexMap_), partial(false), separateMaster(false) {
	// std::cout << "FacilityLocation Clustered Constructor\n";
	//n = n_;
	//mode = clustered;
	//num_clusters = clusters_.size();
	//clusters = clusters_;
	//clusterKernels = clusterKernels_;
	//clusterIndexMap = clusterIndexMap_;
	//partial = false;
	//separateMaster = false;

	//create groundSet with items 0 to nv-1
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i) {
		effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();

	n_master=numEffectiveGroundset;
	masterSet=effectiveGroundSet;

	clusterIDs.resize(n);
	for(int i=0;i<num_clusters;++i) {  //O(n) (One time operation)
		std::unordered_set<ll> ci=clusters[i];
		//for (auto it = ci.begin(); it != ci.end(); ++it) {
		for (auto ind: ci) {
			//ll ind = *it;
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

FacilityLocation::FacilityLocation(const FacilityLocation& f)
    : n(f.n),
	n_master(f.n_master),
	mode(f.mode),
    denseKernel(f.denseKernel),
    partial(f.partial),
	separateMaster(f.separateMaster),
    effectiveGroundSet(f.effectiveGroundSet),
	masterSet(f.masterSet),
	numEffectiveGroundset(f.numEffectiveGroundset),
	originalToPartialIndexMap(f.originalToPartialIndexMap),
	sparseKernel(f.sparseKernel),
	num_clusters(f.num_clusters),
	clusters(f.clusters),
	clusterIDs(f.clusterIDs),
	clusterKernels(f.clusterKernels),
	clusterIndexMap(f.clusterIndexMap),
	similarityWithNearestInEffectiveX(f.similarityWithNearestInEffectiveX),
	relevantX(f.relevantX),
	clusteredSimilarityWithNearestInRelevantX(f.clusteredSimilarityWithNearestInRelevantX) {

	}

FacilityLocation* FacilityLocation::clone() {
    return new FacilityLocation(*this);
}

//helper friend function
// float get_max_sim_dense(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj) {
// 	if(dataset_ind.size()==0) {
// 		return 0;
// 	}
// 	ll i = datapoint_ind, j; //i comes from masterSet and j comes from X (which is a subset of groundSet)
// 	auto it = dataset_ind.begin();
// 	float m = obj.denseKernel[i][*it];

// 	for (; it != dataset_ind.end(); ++it) {//search max similarity wrt datapoints of given dataset
// 		ll j = *it;
// 		if (obj.denseKernel[i][j] > m) {
// 			m = obj.denseKernel[i][j];
// 		}
// 	}

// 	return m;
// }

float get_max_sim_dense(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj) {
	float m = 0;
    for(auto elem: dataset_ind) {
		if(obj.denseKernel[datapoint_ind][elem] > m) {
			m = obj.denseKernel[datapoint_ind][elem];
		}
	}
	return m;
}

// float get_max_sim_sparse(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj) {
// 	if(dataset_ind.size()==0) {
// 		return 0;
// 	}
// 	ll i = datapoint_ind, j; //i comes from masterSet and j comes from X (which is a subset of groundSet)
// 	auto it = dataset_ind.begin();
// 	float m = obj.sparseKernel.get_val(i,*it);
	
// 	for (; it != dataset_ind.end(); ++it) {//search max similarity wrt datapoints of given dataset
// 		ll j = *it;
// 		float temp_val = obj.sparseKernel.get_val(i,j);
// 		if (temp_val > m) {
// 			m = temp_val;
// 		}
// 	}

// 	return m;
// }

float get_max_sim_sparse(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj) {
	float m = 0;
	for(auto elem: dataset_ind) {
		float temp = obj.sparseKernel.get_val(datapoint_ind, elem);
		if(temp > m) {
			m = temp;
		}
	}
	return m;
}

// float get_max_sim_cluster(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj, ll cluster_id) {
//     if(dataset_ind.size()==0) {
// 		return 0;
// 	}
// 	ll i = datapoint_ind, j, i_, j_; 
// 	auto it = dataset_ind.begin();
// 	i_ = obj.clusterIndexMap[i];
// 	j_ = obj.clusterIndexMap[*it]; 
// 	float m = obj.clusterKernels[cluster_id][i_][j_];
// 	//Possibly transform i,j to local kernel index

// 	for (; it != dataset_ind.end(); ++it) {
// 		ll j = *it;
// 		//Obtain local kernel indicies for given cluster
// 		i_ = obj.clusterIndexMap[i];
// 		j_ = obj.clusterIndexMap[j];
// 		if (obj.clusterKernels[cluster_id][i_][j_] > m) {
// 			m = obj.clusterKernels[cluster_id][i_][j_];
// 		}
// 	}
// 	return m;
// }

float get_max_sim_cluster(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj, ll cluster_id) {
    float m = 0;
	ll datapoint_ind_ = obj.clusterIndexMap[datapoint_ind];
	for(auto elem: dataset_ind) {
        ll elem_ = obj.clusterIndexMap[elem];
		if(obj.clusterKernels[cluster_id][datapoint_ind_][elem_] > m) {
			m = obj.clusterKernels[cluster_id][datapoint_ind_][elem_];
		}
	}
	return m;
}

double FacilityLocation::evaluate(std::unordered_set<ll> const &X) {
	// std::cout << "FacilityLocation evaluate\n";
	std::unordered_set<ll> effectiveX;
	double result=0;

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

	if (mode == dense) {
		//for each element in master set
		//for (auto it = masterSet.begin(); it != masterSet.end(); ++it) { //O(n^2) where n=num of elements in effective GS 
		for (auto ind: masterSet) { //O(n^2) where n=num of elements in effective GS 
			//ll ind = *it;
			//find max similarity of i with all items in effectiveX
			result += get_max_sim_dense(ind, effectiveX, *this);
		}
	} else if (mode == sparse) {
        //for (auto it = masterSet.begin(); it != masterSet.end(); ++it) { //O(n^2) where n=num of elements in effective GS 
		for (auto ind: masterSet) { //O(n^2) where n=num of elements in effective GS 
				//ll ind = *it;
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
			
			//for (auto it = ci.begin(); it != ci.end(); ++it) {
			for (auto ind: ci) {
				//ll ind = *it;
				result += get_max_sim_cluster(ind, releventSubset, *this, i);
			}
		}
	}
	return result;
}


double FacilityLocation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	// std::cout << "FacilityLocation evaluateWithMemoization\n";
    //assumes that appropriate pre computed memoized statistics exist for effectiveX

	std::unordered_set<ll> effectiveX;
	double result = 0;

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

	if (mode == dense || mode == sparse) {
		//for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
		for (auto ind: masterSet) {
			//ll ind = *it;
			result += similarityWithNearestInEffectiveX[(partial)?originalToPartialIndexMap[ind]:ind];
		}
	} else {
		for(ll i=0;i<num_clusters;++i) {
			if(relevantX[i].size()==0) {
				continue;
			}
			std::unordered_set<ll> ci = clusters[i];
			//for (auto it = ci.begin(); it != ci.end(); ++it) {
			for (auto ind: ci) {
				//ll ind = *it;
				result += clusteredSimilarityWithNearestInRelevantX[ind];
			}
		}
	}
	return result;
}


double FacilityLocation::marginalGain(std::unordered_set<ll> const &X, ll item) {
	// std::cout << "FacilityLocation marginalGain\n";
	std::unordered_set<ll> effectiveX;
	double gain = 0;

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

	if (effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return 0;
    }

	if (mode == dense) {
		//for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
		for (auto ind: masterSet) {
			//ll ind = *it;
			float m = get_max_sim_dense(ind, effectiveX, *this);
			if (denseKernel[ind][item] > m) {
				gain += (denseKernel[ind][item] - m);
			}
		}
	} else if (mode == sparse) {
			//for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
			for (auto ind: masterSet) {
				//ll ind = *it;
				float m = get_max_sim_sparse(ind, effectiveX, *this);
				float temp = sparseKernel.get_val(ind,item);
				if (temp > m) {
					gain += (temp - m);
				}
			}
	} else {
        ll i = clusterIDs[item];
		ll item_ = clusterIndexMap[item];
		std::unordered_set<ll> releventSubset;
		std::unordered_set<ll> ci = clusters[i];
		// std::set_intersection(X.begin(), X.end(), ci.begin(), ci.end(), std::inserter(releventSubset, releventSubset.begin()));
		releventSubset = set_intersection(X, ci);
		if(releventSubset.size()==0) {
			//for (auto it = ci.begin(); it != ci.end(); ++it) {
			for (auto ind: ci) {
				//ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				gain+=clusterKernels[i][ind_][item_];
			}
		} else {
			//for (auto it = ci.begin(); it != ci.end(); ++it) {
			for (auto ind: ci) {
				//ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				//ll item_ = clusterIndexMap[item];
				float m = get_max_sim_cluster(ind, releventSubset, *this, i);
				if (clusterKernels[i][ind_][item_] > m) {
					gain += (clusterKernels[i][ind_][item_] - m);
				}
			}
		}
	}
	return gain;
}


double FacilityLocation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	// std::cout << "FacilityLocation marginalGainWithMemoization\n";
	std::unordered_set<ll> effectiveX;
	double gain = 0;
	if (partial == true) {
		// std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	    effectiveX = set_intersection(X, effectiveGroundSet);
	} else {
		effectiveX = X;
	}
	if (enableChecks && effectiveX.find(item)!=effectiveX.end()) {
		//item is already present
		return 0;
	}
	if (partial && (effectiveGroundSet.find(item)==effectiveGroundSet.end())) {
        return 0;
    }
	if (mode == dense) {
		//for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
		if (partial) {
			for (auto ind : masterSet) {
				// ll ind = *it;
				if (denseKernel[ind][item] >
					similarityWithNearestInEffectiveX
						[originalToPartialIndexMap[ind]]) {
					gain +=
						(denseKernel[ind][item] -
							similarityWithNearestInEffectiveX
								[originalToPartialIndexMap[ind]]);
				}
			}
		} else {
			for (auto ind : masterSet) {
				// ll ind = *it;
				if (denseKernel[ind][item] >
					similarityWithNearestInEffectiveX[ind]) {
					gain +=
						(denseKernel[ind][item] -
							similarityWithNearestInEffectiveX[ind]);
				}
			}
		}

	} else if (mode == sparse) {
	    //for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
		for (auto ind: masterSet) {
			//ll ind = *it;
			float temp = sparseKernel.get_val(ind,item);
			if (temp > similarityWithNearestInEffectiveX[ind]) {
				gain += (temp - similarityWithNearestInEffectiveX[ind]);
			}
		}
	} else {
        ll i = clusterIDs[item];
		ll item_ = clusterIndexMap[item];
		std::unordered_set<ll> releventSubset = relevantX[i];
		std::unordered_set<ll> ci = clusters[i];
		
		if(releventSubset.size()==0) {
			//for (auto it = ci.begin(); it != ci.end(); ++it) {
			for (auto ind: ci) {
				//ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				gain+=clusterKernels[i][ind_][item_];
			}
		} else {
			//for (auto it = ci.begin(); it != ci.end(); ++it) {
			for (auto ind: ci) {
				//ll ind = *it;
				ll ind_=clusterIndexMap[ind];
				//ll item_ = clusterIndexMap[item];
				if (clusterKernels[i][ind_][item_] > clusteredSimilarityWithNearestInRelevantX[ind]) {
					gain += (clusterKernels[i][ind_][item_] - clusteredSimilarityWithNearestInRelevantX[ind]);
				}
			}
		}
	}
	return gain;
}

void FacilityLocation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
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
	if (partial && (effectiveGroundSet.find(item)==effectiveGroundSet.end())) {
        return;
    }
	if (mode == dense) {
		//for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
		if(partial) {
			for (auto ind: masterSet) {
				//ll ind = *it;
				if (denseKernel[ind][item] > similarityWithNearestInEffectiveX[originalToPartialIndexMap[ind]]) {
					similarityWithNearestInEffectiveX[originalToPartialIndexMap[ind]] = denseKernel[ind][item];
				}
			}
		} else {
			for (auto ind: masterSet) {
				//ll ind = *it;
				if (denseKernel[ind][item] > similarityWithNearestInEffectiveX[ind]) {
					similarityWithNearestInEffectiveX[ind] = denseKernel[ind][item];
				}
			}
		}
	} else if (mode == sparse) {
		//for (auto it = masterSet.begin(); it != masterSet.end(); ++it) {
		for (auto ind: masterSet) {
			//ll ind = *it;
			float temp_val = sparseKernel.get_val(ind,item);
			if (temp_val > similarityWithNearestInEffectiveX[ind]) {
				similarityWithNearestInEffectiveX[ind] = temp_val;
			}
		}
	} else {
        ll i = clusterIDs[item];
		ll item_ = clusterIndexMap[item];
		std::unordered_set<ll> ci = clusters[i];
		//for (auto it = ci.begin(); it != ci.end(); ++it) {
		for (auto ind: ci) {
			//ll ind = *it;
			ll ind_=clusterIndexMap[ind];
			if (clusterKernels[i][ind_][item_] > clusteredSimilarityWithNearestInRelevantX[ind]) {
				clusteredSimilarityWithNearestInRelevantX[ind]= clusterKernels[i][ind_][item_];
			}		 
		}
		relevantX[i].insert(item);
	}
}

std::unordered_set<ll> FacilityLocation::getEffectiveGroundSet() {
	// std::cout << "FacilityLocation getEffectiveGroundSet\n";
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> FacilityLocation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "FacilityLocation maximize\n";
// 	if(optimizer == "NaiveGreedy") {
// 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "LazyGreedy") {
//         return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "StochasticGreedy") {
//         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else if(optimizer == "LazierThanLazyGreedy") {
//         return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else {
// 		std::cerr << "Invalid Optimizer" << std::endl;
// 	}
// }


void FacilityLocation::cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda) {
	// std::cout << "FacilityLocation clusterInit\n";
	*this = FacilityLocation(n_, denseKernel_, partial, ground_, false);
}

void FacilityLocation::clearMemoization() {
	// std::cout << "FacilityLocation clearMemoization\n";
	//We could do https://stackoverflow.com/questions/55266468/whats-the-fastest-way-to-reinitialize-a-vector/55266856 to replace it with a more efficient implementation. However, clear() and assign() involve re-alloc and hence are slower. fill() also involves a loop. memset could be faster, but being at lower level, could be unsafe to use

    //reset similarityWithNearestInEffectiveX for dense and sparse modes
	if(mode==dense || mode==sparse) {
		for(ll i=0;i<n_master;++i) {
			similarityWithNearestInEffectiveX[i]=0;
		}
	} else {
	    //reset relevantX and clusteredSimilarityWithNearestInRelevantX for clustered mode
		for(int i=0;i<num_clusters;++i) {
			relevantX[i].clear();
		}
		for(ll i=0;i<n;++i) {
			clusteredSimilarityWithNearestInRelevantX[i]=0;
		}
	}
}

void FacilityLocation::setMemoization(std::unordered_set<ll> const &X) 
{
	// std::cout << "FacilityLocation setMemoization\n";
    clearMemoization();
    std::unordered_set<ll> temp;
	//for (auto it = X.begin(); it != X.end(); ++it)
    for (auto elem: X)
	{	
		updateMemoization(temp, elem);
		temp.insert(elem);	
	}
}


