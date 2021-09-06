#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"SetCoverMutualInformation.h"

SetCoverMutualInformation::SetCoverMutualInformation(ll n_, std::vector<std::unordered_set<int>> const &coverSet_, int numConcepts_, std::vector<float> const& conceptWeights_, std::unordered_set<int> const & Q_): n(n_), coverSet(coverSet_), numConcepts(numConcepts_), conceptWeights(conceptWeights_), Q(Q_)  {
	coverSetQ = std::vector<std::unordered_set<int>>();
    for (ll i = 0; i < coverSet.size(); i++) {
        std::unordered_set<int> coverSetQcurrSet;
        coverSetQcurrSet.clear();
        for (auto elem: coverSet[i]) {
            if (Q.find(elem) != Q.end()) coverSetQcurrSet.insert(elem);
        }
        coverSetQ.push_back(coverSetQcurrSet);
    }
    scQ = new SetCover(n, coverSetQ, numConcepts, conceptWeights);
}

double SetCoverMutualInformation::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}
	return scQ->evaluate(X);
}

double SetCoverMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	return scQ->evaluateWithMemoization(X);
}


double SetCoverMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	
	return scQ->marginalGain(X, item);
}


double SetCoverMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	return scQ->marginalGainWithMemoization(X, item);
}

void SetCoverMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	scQ->updateMemoization(X, item);
}

std::unordered_set<ll> SetCoverMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> SetCoverMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void SetCoverMutualInformation::clearMemoization() {
	scQ->clearMemoization();
}

void SetCoverMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    scQ->setMemoization(X);
}


