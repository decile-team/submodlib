#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"ProbabilisticSetCoverConditionalGain.h"

ProbabilisticSetCoverConditionalGain::ProbabilisticSetCoverConditionalGain(ll n_, int numConcepts_,std::vector<std::vector<float>> const &groundSetConceptProbabilities_,  std::vector<float> const& conceptWeights_, std::unordered_set<int> const &P_): n(n_), groundSetConceptProbabilities(groundSetConceptProbabilities_), numConcepts(numConcepts_), conceptWeights(conceptWeights_), P(P_)  {
	conceptWeightsP = std::vector<float>();
    float prob_P_doesnt_cover_i = 1;
    for (int i = 0; i < numConcepts; i++) {
        prob_P_doesnt_cover_i = 1;
        if (P.find(i) != P.end()) prob_P_doesnt_cover_i = 0;
        conceptWeightsP.push_back(conceptWeights[i] * prob_P_doesnt_cover_i);
    }
    pscP =
        new ProbabilisticSetCover(n, groundSetConceptProbabilities, numConcepts, conceptWeightsP);
}

// ProbabilisticSetCoverConditionalGain* ProbabilisticSetCoverConditionalGain::clone() {
// 	return NULL;
// }

double ProbabilisticSetCoverConditionalGain::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}
	return pscP->evaluate(X);
}

double ProbabilisticSetCoverConditionalGain::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	return pscP->evaluateWithMemoization(X);
}


double ProbabilisticSetCoverConditionalGain::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	
	return pscP->marginalGain(X, item);
}


double ProbabilisticSetCoverConditionalGain::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	return pscP->marginalGainWithMemoization(X, item);
}

void ProbabilisticSetCoverConditionalGain::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	pscP->updateMemoization(X, item);
}

std::unordered_set<ll> ProbabilisticSetCoverConditionalGain::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> ProbabilisticSetCoverConditionalGain::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void ProbabilisticSetCoverConditionalGain::clearMemoization() {
	pscP->clearMemoization();
}

void ProbabilisticSetCoverConditionalGain::setMemoization(std::unordered_set<ll> const &X) 
{
    pscP->setMemoization(X);
}


