#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"ProbabilisticSetCover.h"

ProbabilisticSetCover::ProbabilisticSetCover(ll n_, std::vector<std::vector<float>> const &groundSetConceptProbabilities_, int numConcepts_, std::vector<float> const& conceptWeights_): n(n_), groundSetConceptProbabilities(groundSetConceptProbabilities_), numConcepts(numConcepts_), conceptWeights(conceptWeights_)  {
	probOfConceptsCoveredByX = std::vector<double>(numConcepts, 1);
}

// ProbabilisticSetCover* ProbabilisticSetCover::clone() {
// 	return NULL;
// }

double ProbabilisticSetCover::evaluate(std::unordered_set<ll> const &X) {
	double result=0;

	if(X.size()==0) {
		return 0;
	}

	for(int i=0; i < numConcepts; i++) {
		double product = 1;
		for(auto elem: X) {
			product *= (1-groundSetConceptProbabilities[elem][i]);
		}
		result += conceptWeights[i]*(1-product);
	}
	return result;
}

double ProbabilisticSetCover::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
	double result = 0;
	if(X.size()==0) {
		return 0;
	}
	for(int i=0; i < numConcepts; i++) {
		result += conceptWeights[i]*(1-probOfConceptsCoveredByX[i]);
	}
	return result;
}


double ProbabilisticSetCover::marginalGain(std::unordered_set<ll> const &X, ll item) {
	double gain = 0;
	if (X.find(item)!=X.end()) {
		return 0;
	}
	
	for(int i=0; i<numConcepts; i++) {
		double oldConceptProd = 1;
		for(auto elem: X) {
			oldConceptProd *= (1-groundSetConceptProbabilities[elem][i]);
		}
		gain += conceptWeights[i]*oldConceptProd*groundSetConceptProbabilities[item][i];
	}
	return gain;
}


double ProbabilisticSetCover::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
	double gain = 0;
	if (enableChecks && X.find(item)!=X.end()) {
		return 0;
	}
	for(int i=0; i<numConcepts; i++) {
		gain += conceptWeights[i]*probOfConceptsCoveredByX[i]*groundSetConceptProbabilities[item][i];
	}
	return gain;
}

void ProbabilisticSetCover::updateMemoization(std::unordered_set<ll> const &X, ll item) {
	if (X.find(item)!=X.end()) {
		return;
	}
	for(int i=0; i<numConcepts; i++) {
		probOfConceptsCoveredByX[i] *= (1-groundSetConceptProbabilities[item][i]);
	}
}

std::unordered_set<ll> ProbabilisticSetCover::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}


// std::vector<std::pair<ll, double>> ProbabilisticSetCover::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
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

void ProbabilisticSetCover::clearMemoization() {
	for(int i=0; i<numConcepts; i++) {
		probOfConceptsCoveredByX[i] = 1;
	}
}

void ProbabilisticSetCover::setMemoization(std::unordered_set<ll> const &X) 
{
    clearMemoization();
    std::unordered_set<ll> temp;
    for (auto elem: X)
	{	
		updateMemoization(temp, elem);
		temp.insert(elem);	
	}
}


