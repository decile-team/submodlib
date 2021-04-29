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
		for(auto concept: coverSet[elem]) {
			conceptsCovered.insert(concept);
		}
	}
	for(auto concept: conceptsCovered) {
		result += conceptWeights[concept];
	}

	return result;
}

double SetCover::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	
	for(auto concept: conceptsCoveredByX) {
		result += conceptWeights[concept];
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
		for(auto concept: coverSet[elem]) {
			conceptsCovered.insert(concept);
		}
	}
	for(auto concept: coverSet[item]) {
        if(conceptsCovered.find(concept) == conceptsCovered.end()) {
            gain += conceptWeights[concept];
		}
	}
	return gain;
}


double SetCover::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	//std::cout << "Concepts covered by " << item << " = {";
	for(auto concept: coverSet[item]) {
		//std::cout << concept << ", ";
        if(conceptsCoveredByX.find(concept) == conceptsCoveredByX.end()) {
            gain += conceptWeights[concept];
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
	for(auto concept: coverSet[item]) {
		conceptsCoveredByX.insert(concept);
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


std::vector<std::pair<ll, double>> SetCover::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false) {
	if(optimizer == "NaiveGreedy") {
		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose);
	} else if(optimizer == "LazyGreedy") {
        return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose);
	} else if(optimizer == "StochasticGreedy") {
        return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose);
	} else if(optimizer == "LazierThanLazyGreedy") {
        return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose);
	} else {
		throw "Error: Invalid Optimizer";
	}
}

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


