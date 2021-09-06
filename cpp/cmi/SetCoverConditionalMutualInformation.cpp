#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"SetCoverConditionalMutualInformation.h"

SetCoverConditionalMutualInformation::SetCoverConditionalMutualInformation(ll n_, std::vector<std::unordered_set<int>> const &coverSet_, int numConcepts_, std::vector<float> const& conceptWeights_, std::unordered_set<int> const & Q_, std::unordered_set<int> const & P_): n(n_), coverSet(coverSet_), numConcepts(numConcepts_), conceptWeights(conceptWeights_), Q(Q_), P(P_)  {
	coverSetQMinusP = std::vector<std::unordered_set<int>>();
    for (ll i = 0; i < coverSet.size(); i++) {
        std::unordered_set<int> coverSetQMinusPcurrSet;
        coverSetQMinusPcurrSet.clear();
        for (auto elem: coverSet[i]) {
            if (Q.find(elem) != Q.end() && P.find(elem) == P.end()) coverSetQMinusPcurrSet.insert(elem);
        }
        coverSetQMinusP.push_back(coverSetQMinusPcurrSet);
    }
    scQMinusP = new SetCover(n, coverSetQMinusP, numConcepts, conceptWeights);
}

double SetCoverConditionalMutualInformation::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}
	return scQMinusP->evaluate(X);
}

double SetCoverConditionalMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	return scQMinusP->evaluateWithMemoization(X);
}


double SetCoverConditionalMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	
	return scQMinusP->marginalGain(X, item);
}


double SetCoverConditionalMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	return scQMinusP->marginalGainWithMemoization(X, item);
}

void SetCoverConditionalMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	scQMinusP->updateMemoization(X, item);
}

std::unordered_set<ll> SetCoverConditionalMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> SetCoverConditionalMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void SetCoverConditionalMutualInformation::clearMemoization() {
	scQMinusP->clearMemoization();
}

void SetCoverConditionalMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    scQMinusP->setMemoization(X);
}


