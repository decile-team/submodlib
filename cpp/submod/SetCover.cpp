#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"SetCover.h"

SetCover::SetCover(ll n_, std::vector<std::unordered_set<int>> const &coverSet_, int numConcepts_, std::vector<float> const& conceptWeights_): n(n_), coverSet(coverSet_), numConcepts(numConcepts_), conceptWeights(conceptWeights_)  {
}

// SetCover* SetCover::clone() {
// 	return NULL;
// }

double SetCover::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}

	std::unordered_set<int> conceptsCovered;
	for(auto elem: X) {
		for(auto con: coverSet[elem]) {
			conceptsCovered.insert(con);
		}
	}
	for(auto con: conceptsCovered) {
		result += conceptWeights[con];
	}

	return result;
}

double SetCover::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	
	for(auto con: conceptsCoveredByX) {
		result += conceptWeights[con];
	}
	return result;
}


double SetCover::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
    std::unordered_set<int> conceptsCovered;
	for(auto elem: X) {
		for(auto con: coverSet[elem]) {
			conceptsCovered.insert(con);
		}
	}
	for(auto con: coverSet[item]) {
        if(conceptsCovered.find(con) == conceptsCovered.end()) {
            gain += conceptWeights[con];
		}
	}
	return gain;
}


double SetCover::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	//std::cout << "Concepts covered by " << item << " = {";
	for(auto con: coverSet[item]) {
		//std::cout << concept << ", ";
        if(conceptsCoveredByX.find(con) == conceptsCoveredByX.end()) {
            gain += conceptWeights[con];
		}
	}
	//std::cout << "}\n";
	//std::cout << "Gain = " << gain << "\n";
	return gain;
}

void SetCover::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	for(auto con: coverSet[item]) {
		conceptsCoveredByX.insert(con);
	}
}

std::unordered_set<ll> SetCover::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> SetCover::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void SetCover::clearMemoization() {
	conceptsCoveredByX.clear();
}

void SetCover::setMemoization(std::unordered_set<ll> const &X) 
{
    clearMemoization();
    std::unordered_set<ll> temp;
    for (auto elem: X)
	{	
		updateMemoization(temp, elem);
		temp.insert(elem);	
	}
}


