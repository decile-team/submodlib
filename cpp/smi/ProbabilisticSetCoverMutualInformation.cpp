#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"ProbabilisticSetCoverMutualInformation.h"

ProbabilisticSetCoverMutualInformation::ProbabilisticSetCoverMutualInformation(ll n_, int numConcepts_,std::vector<std::vector<float>> const &groundSetConceptProbabilities_,  std::vector<float> const& conceptWeights_, std::unordered_set<int> const &Q_): n(n_), groundSetConceptProbabilities(groundSetConceptProbabilities_), numConcepts(numConcepts_), conceptWeights(conceptWeights_), Q(Q_)  {
	conceptWeightsQ = std::vector<float>();
    float prob_Q_covers_i = 0;
    for (int i = 0; i < numConcepts; i++) {
        prob_Q_covers_i = 0;
        if (Q.find(i) != Q.end()) prob_Q_covers_i = 1;
        conceptWeightsQ.push_back(conceptWeights[i] * prob_Q_covers_i);
    }
    pscQ =
        new ProbabilisticSetCover(n, groundSetConceptProbabilities, numConcepts, conceptWeightsQ);
}

// ProbabilisticSetCoverMutualInformation* ProbabilisticSetCoverMutualInformation::clone() {
// 	return NULL;
// }

double ProbabilisticSetCoverMutualInformation::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}
	return pscQ->evaluate(X);
}

double ProbabilisticSetCoverMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	return pscQ->evaluateWithMemoization(X);
}


double ProbabilisticSetCoverMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	
	return pscQ->marginalGain(X, item);
}


double ProbabilisticSetCoverMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	return pscQ->marginalGainWithMemoization(X, item);
}

void ProbabilisticSetCoverMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	pscQ->updateMemoization(X, item);
}

std::unordered_set<ll> ProbabilisticSetCoverMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> ProbabilisticSetCoverMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void ProbabilisticSetCoverMutualInformation::clearMemoization() {
	pscQ->clearMemoization();
}

void ProbabilisticSetCoverMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    pscQ->setMemoization(X);
}


