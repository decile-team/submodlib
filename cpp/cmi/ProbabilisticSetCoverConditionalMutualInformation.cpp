#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"ProbabilisticSetCoverConditionalMutualInformation.h"

ProbabilisticSetCoverConditionalMutualInformation::ProbabilisticSetCoverConditionalMutualInformation(ll n_, int numConcepts_,std::vector<std::vector<float>> const &groundSetConceptProbabilities_,  std::vector<float> const& conceptWeights_, std::unordered_set<int> const &Q_, std::unordered_set<int> const &P_): n(n_), groundSetConceptProbabilities(groundSetConceptProbabilities_), numConcepts(numConcepts_), conceptWeights(conceptWeights_), Q(Q_), P(P_)  {
	conceptWeightsQAndP = std::vector<float>();
    float prob_Q_covers_i = 0;
	float prob_P_doesnt_cover_i = 1;
    for (int i = 0; i < numConcepts; i++) {
        prob_Q_covers_i = 0;
        if (Q.find(i) != Q.end()) prob_Q_covers_i = 1;
		prob_P_doesnt_cover_i = 1;
        if (P.find(i) != P.end()) prob_P_doesnt_cover_i = 0;
        conceptWeightsQAndP.push_back(conceptWeights[i] * prob_Q_covers_i * prob_P_doesnt_cover_i);
    }
    pscQAndP =
        new ProbabilisticSetCover(n, groundSetConceptProbabilities, numConcepts, conceptWeightsQAndP);
}

// ProbabilisticSetCoverConditionalMutualInformation* ProbabilisticSetCoverConditionalMutualInformation::clone() {
// 	return NULL;
// }

double ProbabilisticSetCoverConditionalMutualInformation::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}
	return pscQAndP->evaluate(X);
}

double ProbabilisticSetCoverConditionalMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	return pscQAndP->evaluateWithMemoization(X);
}


double ProbabilisticSetCoverConditionalMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	
	return pscQAndP->marginalGain(X, item);
}


double ProbabilisticSetCoverConditionalMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	return pscQAndP->marginalGainWithMemoization(X, item);
}

void ProbabilisticSetCoverConditionalMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	pscQAndP->updateMemoization(X, item);
}

std::unordered_set<ll> ProbabilisticSetCoverConditionalMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> ProbabilisticSetCoverConditionalMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void ProbabilisticSetCoverConditionalMutualInformation::clearMemoization() {
	pscQAndP->clearMemoization();
}

void ProbabilisticSetCoverConditionalMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    pscQAndP->setMemoization(X);
}


