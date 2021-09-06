#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"GraphCutMutualInformation.h"

GraphCutMutualInformation::GraphCutMutualInformation(ll n_, int numQueries_, std::vector<std::vector<float>> const &kernelQuery_): n(n_), numQueries(numQueries_), kernelQuery(kernelQuery_){
    qSum.clear();
    for (ll i = 0; i < n; i++) {
        double sum = 0;
        for(int q = 0; q < numQueries; q++) {
            sum += kernelQuery[i][q];
        }
        qSum.push_back(sum);
    }
    evalX = 0;
}

// GraphCutMutualInformation* GraphCutMutualInformation::clone() {
//     return NULL;
// }

double GraphCutMutualInformation::evaluate(std::unordered_set<ll> const &X) {
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }
    for (auto elem: X) {
        result += qSum[elem];
    }
    return result;
}

double GraphCutMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    return evalX;
}

double GraphCutMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
        return 0;
    }
    double gain = 0;
    for(int q=0; q<numQueries; q++) {
        gain += kernelQuery[item][q];
    }
    return gain;
}

double GraphCutMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }
    double gain = 0;
    for(int q=0; q<numQueries; q++) {
        gain += kernelQuery[item][q];
    }
    return gain;
}

void GraphCutMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    for (int q = 0; q < numQueries; q++) {
        evalX += kernelQuery[item][q];
    }
}

// std::vector<std::pair<ll, double>> GraphCutMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "GraphCutMutualInformation maximize\n";
// 	if(optimizer == "NaiveGreedy") {
// 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "LazyGreedy") {
//         return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "StochasticGreedy") {
//         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else if(optimizer == "LazierThanLazyGreedy") {
//         return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else {
// 		std::cerr << "Invalid Optimizer" << std::endl;
// 	}
// }

std::unordered_set<ll> GraphCutMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void GraphCutMutualInformation::clearMemoization()
{
    evalX = 0;
}

void GraphCutMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    evalX = evaluate(X);
}
