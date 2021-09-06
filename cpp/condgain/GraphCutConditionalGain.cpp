#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"GraphCutConditionalGain.h"

GraphCutConditionalGain::GraphCutConditionalGain(ll n_, int numPrivates_, std::vector<std::vector<float>> const &kernelImage_, std::vector<std::vector<float>> const &kernelPrivate_, float privacyHardness_, float lambda_): n(n_), numPrivates(numPrivates_), kernelImage(kernelImage_), kernelPrivate(kernelPrivate_), privacyHardness(privacyHardness_), lambda(lambda_) {
    totalSimilarityWithSubset.resize(n);
	totalSimilarityWithMaster.resize(n);
	for(ll i=0; i<n; i++) {
		totalSimilarityWithSubset[i] = 0;
        totalSimilarityWithMaster[i] = 0;
		for(ll j=0; j<n; j++) {
			totalSimilarityWithMaster[i] += kernelImage[j][i];
		}
	}
    pSum.clear();
    for (ll i = 0; i < n; i++) {
        double sum = 0;
        for(int p = 0; p < numPrivates; p++) {
            sum += kernelPrivate[i][p];
        }
        pSum.push_back(sum * privacyHardness);
    }
}

// GraphCutConditionalGain* GraphCutConditionalGain::clone() {
//     return NULL;
// }

double GraphCutConditionalGain::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "GraphCutConditionalGain's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    //regular GC
    for(auto elem: X) {
        result += totalSimilarityWithMaster[elem];
        for(auto elem2: X) {
            result -= lambda * kernelImage[elem][elem2];
        }
        result -= 2*lambda*pSum[elem];
    }
    return result;
}

double GraphCutConditionalGain::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    for(auto elem: X) {
        result += totalSimilarityWithMaster[elem] - lambda * totalSimilarityWithSubset[elem] - 2*lambda*pSum[elem];
    }
    return result;
}

double GraphCutConditionalGain::marginalGain(std::unordered_set<ll> const &X, ll item) {
    double gain = 0;

    if (X.find(item)!=X.end()) {
        return 0;
    }

    for (ll i = 0; i < n; i++) {
        gain += kernelImage[i][item];
    }
    for (auto it: X) {
        gain -= lambda * 2 * kernelImage[it][item];
    }
    gain -= lambda * kernelImage[item][item];
    gain -= 2*lambda*pSum[item];
    return gain;
}

double GraphCutConditionalGain::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }

    for (ll i = 0; i < n; i++) {
        gain += kernelImage[i][item];
    }
    for (auto it: X) {
        gain -= lambda * 2 * kernelImage[it][item];
    }
    gain -= lambda * kernelImage[item][item];
    gain -= 2*lambda*pSum[item];
    return gain;
}

void GraphCutConditionalGain::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    for(ll elem=0; elem<n; elem++) 
        totalSimilarityWithSubset[elem] += kernelImage[elem][item];
}

// std::vector<std::pair<ll, double>> GraphCutConditionalGain::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "GraphCutConditionalGain maximize\n";
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

std::unordered_set<ll> GraphCutConditionalGain::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void GraphCutConditionalGain::clearMemoization()
{
    for (ll i = 0; i < n; i++) {
        totalSimilarityWithSubset[i] = 0;
    }
}

void GraphCutConditionalGain::setMemoization(std::unordered_set<ll> const &X) 
{
    clearMemoization();
    std::unordered_set<ll> temp;
	//for (auto it = X.begin(); it != X.end(); ++it)
    for (auto elem: X)
	{	
		updateMemoization(temp, elem);
		temp.insert(elem);	
	}
}
