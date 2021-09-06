#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"GraphCut.h"

GraphCut::GraphCut(){}

//Constructor for dense mode with master = ground
GraphCut::GraphCut(ll n_, std::vector<std::vector<float>> const &masterGroundKernel_, bool partial_, std::unordered_set<ll> const &ground_, float lambda_): n(n_), mode(dense), masterGroundKernel(masterGroundKernel_), partial(partial_), separateMaster(false), lambda(lambda_), groundGroundKernel(masterGroundKernel_) {
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
	
	//master set will now be same as the ground set
	n_master=numEffectiveGroundset;
	masterSet=effectiveGroundSet;

	if(partial == true) {
		ll ind = 0;
		//for (auto it = effectiveGroundSet.begin(); it != effectiveGroundSet.end(); ++it) {
		for (auto it: effectiveGroundSet) {
			originalToPartialIndexMap[it] = ind;
			ind += 1;
		}
		// std::cout << "originalToPartialIndexMap = {";
		// for (auto it: effectiveGroundSet) {
		// 	std::cout << it << ":" << originalToPartialIndexMap[it] <<", ";
		// }
		// std::cout << "}\n";
	}
	
	
	totalSimilarityWithSubset.resize(numEffectiveGroundset);
	totalSimilarityWithMaster.resize(numEffectiveGroundset);
	for(auto elem: effectiveGroundSet) {
		totalSimilarityWithSubset[(partial)?originalToPartialIndexMap[elem]:elem] = 0;
        totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[elem]:elem] = 0;
		for(auto j: masterSet) {
			totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[elem]:elem] += masterGroundKernel[j][elem];
		}
	}
	// std::cout << "Effective ground set: {";
	// for(auto elem: effectiveGroundSet) {
	// 	std::cout << elem << ", ";
	// }
	// std::cout << "}\n";

	// std::cout << "Total similarity with master: {";
	// for(auto elem: totalSimilarityWithMaster) {
	// 	std::cout << elem << ", ";
	// }
	// std::cout << "}\n";
}

// GraphCut* GraphCut::clone() {
// 	return NULL;
// }

//Constructor for dense mode with separateMaster
GraphCut::GraphCut(ll n_, std::vector<std::vector<float>> const &masterGroundKernel_, std::vector<std::vector<float>> const &groundGroundKernel_, float lambda_): n(n_), mode(dense), masterGroundKernel(masterGroundKernel_), groundGroundKernel(groundGroundKernel_), partial(false), separateMaster(true), lambda(lambda_) {
	
	//create groundSet with items 0 to n-1
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); //each insert takes O(1) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();
	
	//populate a different master set
	n_master = masterGroundKernel.size();	
	masterSet.reserve(n_master);
	for (ll i = 0; i < n_master; ++i) {
		masterSet.insert(i); //each insert takes O(1) time
	}
	
	totalSimilarityWithSubset.resize(n);
	totalSimilarityWithMaster.resize(n);
	for (ll i = 0; i < n; i++) { 
		totalSimilarityWithSubset[i] = 0; 
		totalSimilarityWithMaster[i] = 0;
		for(ll j = 0; j < n_master; j++){
			totalSimilarityWithMaster[i] +=masterGroundKernel[j][i];
		}
	}
}

//For sparse mode
GraphCut::GraphCut(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col, float lambda_): n(n_), mode(sparse), partial(false), separateMaster(false), lambda(lambda_) {
	if (arr_val.size() == 0 || arr_count.size() == 0 || arr_col.size() == 0) {
		throw "Error: Empty/Corrupt sparse similarity kernel";
	}
	sparseKernel = SparseSim(arr_val, arr_count, arr_col);
	//create groundSet with items 0 to nv-1
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i) {
		effectiveGroundSet.insert(i); //each insert takes O(1) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();
	
	n_master=numEffectiveGroundset;
	masterSet=effectiveGroundSet;
    
	totalSimilarityWithSubset.resize(n);
	totalSimilarityWithMaster.resize(n);
	for (ll i = 0; i < n; i++) { 
		totalSimilarityWithSubset[i] = 0; 
		totalSimilarityWithMaster[i] = 0;
		for(ll j = 0; j < n; j++){
			totalSimilarityWithMaster[i] += sparseKernel.get_val(j, i);
		}
	}
}

double GraphCut::evaluate(std::unordered_set<ll> const &X) {
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
		for(auto elem: effectiveX) {
			result += totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[elem]:elem];
			for(auto elem2: effectiveX) {
				result -= lambda * groundGroundKernel[elem][elem2];
			}
		}
	} else if (mode == sparse) {
        for(auto elem: effectiveX) {
			result += totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[elem]:elem];
			for(auto elem2: effectiveX) {
				result -= lambda * sparseKernel.get_val(elem, elem2);
			}
		}
	} else {
		throw "Error: Only dense and sparse mode supported";
	}
	return result;
}


double GraphCut::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
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
		for(auto elem: effectiveX) {
			result += totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[elem]:elem] - lambda * totalSimilarityWithSubset[(partial)?originalToPartialIndexMap[elem]:elem];
		}
	} else {
		throw "Error: Only dense and sparse mode supported";
	}
	return result;
}


double GraphCut::marginalGain(std::unordered_set<ll> const &X, ll item) {
	std::unordered_set<ll> effectiveX;
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

	double gain = totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[item]:item];

	if (mode == dense) {
		for(auto elem: effectiveX) {
			gain -= 2 * lambda * groundGroundKernel[item][elem];
		}
		gain -= lambda * groundGroundKernel[item][item];
	} else if (mode == sparse) {
		for(auto elem: effectiveX) {
			gain -= 2 * lambda * sparseKernel.get_val(item, elem);
		}
		gain -= lambda * sparseKernel.get_val(item, item);
	} else {
        throw "Error: Only dense and sparse mode supported";
	}
	return gain;
}


double GraphCut::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	// std::cout << "GraphCut marginalGainWithMemoization\n";
	std::unordered_set<ll> effectiveX;
	double gain = 0;
	if (partial == true) {
		// std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	    effectiveX = set_intersection(X, effectiveGroundSet);
	} else {
		effectiveX = X;
	}
	if (enableChecks && effectiveX.find(item)!=effectiveX.end()) {
		return 0;
	}
	if (partial && effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return 0;
    }
	if (mode == dense) {
		gain = totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[item]:item] - 2 * lambda * totalSimilarityWithSubset[(partial)?originalToPartialIndexMap[item]:item] - lambda * groundGroundKernel[item][item];
	} else if (mode == sparse) {
		gain = totalSimilarityWithMaster[(partial)?originalToPartialIndexMap[item]:item] - 2 * lambda * totalSimilarityWithSubset[(partial)?originalToPartialIndexMap[item]:item] - lambda * sparseKernel.get_val(item, item);
	} else {
		throw "Error: Only dense and sparse mode supported";
	}
	return gain;
}

void GraphCut::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	// std::cout << "GraphCut updateMemoization\n";
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
	if (effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return;
    }
	if (mode == dense) {
		for(auto elem: effectiveGroundSet) 
			totalSimilarityWithSubset[(partial)?originalToPartialIndexMap[elem]:elem] += groundGroundKernel[elem][item];
	} else if (mode == sparse) {
		for(auto elem: effectiveGroundSet)
			totalSimilarityWithSubset[(partial)?originalToPartialIndexMap[elem]:elem] += sparseKernel.get_val(elem, item);
	} else {
        throw "Error: Only dense and sparse mode supported";
	}
}

std::unordered_set<ll> GraphCut::getEffectiveGroundSet() {
	// std::cout << "GraphCut getEffectiveGroundSet\n";
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> GraphCut::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "GraphCut maximize\n";
// 	if(optimizer == "NaiveGreedy") {
// 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "LazyGreedy") {
//         return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "StochasticGreedy") {
//         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else if(optimizer == "LazierThanLazyGreedy") {
//         return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else {
// 		throw "Error: Invalid Optimizer";
// 	}
// }


void GraphCut::cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda_) {
	// std::cout << "GraphCut clusterInit\n";
	*this = GraphCut(n_, denseKernel_, partial, ground_, lambda_);
}

void GraphCut::clearMemoization() {
	if(mode==dense || mode==sparse) {
		for(ll i=0;i<numEffectiveGroundset;++i) {
			totalSimilarityWithSubset[i]=0;
		}
	} else {
	    throw "Error: Only dense and sparse mode supported";
	}
}

void GraphCut::setMemoization(std::unordered_set<ll> const &X) 
{
	// std::cout << "GraphCut setMemoization\n";
    clearMemoization();
    std::unordered_set<ll> temp;
	//for (auto it = X.begin(); it != X.end(); ++it)
    for (auto elem: X)
	{	
		updateMemoization(temp, elem);
		temp.insert(elem);	
	}
}


