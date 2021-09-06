#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"SetCoverConditionalGain.h"

SetCoverConditionalGain::SetCoverConditionalGain(ll n_, std::vector<std::unordered_set<int>> const &coverSet_, int numConcepts_, std::vector<float> const& conceptWeights_, std::unordered_set<int> const & P_): n(n_), coverSet(coverSet_), numConcepts(numConcepts_), conceptWeights(conceptWeights_), P(P_)  {
	coverSetMinusP = std::vector<std::unordered_set<int>>();
    for (ll i = 0; i < coverSet.size(); i++) {
        std::unordered_set<int> coverSetMinusPcurrSet;
        coverSetMinusPcurrSet.clear();
        for (auto elem: coverSet[i]) {
            if (P.find(elem) == P.end()) coverSetMinusPcurrSet.insert(elem);
        }
        coverSetMinusP.push_back(coverSetMinusPcurrSet);
    }
    scMinusP = new SetCover(n, coverSetMinusP, numConcepts, conceptWeights);
}

double SetCoverConditionalGain::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}
	return scMinusP->evaluate(X);
}

double SetCoverConditionalGain::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	return scMinusP->evaluateWithMemoization(X);
}


double SetCoverConditionalGain::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	
	return scMinusP->marginalGain(X, item);
}


double SetCoverConditionalGain::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	return scMinusP->marginalGainWithMemoization(X, item);
}

void SetCoverConditionalGain::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	scMinusP->updateMemoization(X, item);
}

std::unordered_set<ll> SetCoverConditionalGain::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> SetCoverConditionalGain::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void SetCoverConditionalGain::clearMemoization() {
	scMinusP->clearMemoization();
}

void SetCoverConditionalGain::setMemoization(std::unordered_set<ll> const &X) 
{
    scMinusP->setMemoization(X);
}


