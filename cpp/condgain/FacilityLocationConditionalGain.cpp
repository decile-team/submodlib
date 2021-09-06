#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"FacilityLocationConditionalGain.h"

FacilityLocationConditionalGain::FacilityLocationConditionalGain(ll n_, int numPrivates_, std::vector<std::vector<float>> const &kernelImage_, std::vector<std::vector<float>> const &kernelPrivate_, float privacyHardness_): n(n_), numPrivates(numPrivates_), kernelImage(kernelImage_), kernelPrivate(kernelPrivate_), privacyHardness(privacyHardness_){
    if(privacyHardness != 1) {
        for(ll i=0; i<n; i++) {
            for(int j=0; j<numPrivates; j++) {
                kernelPrivate[i][j] *= privacyHardness;
            }
        }
    }
    std::vector<float> tempVector = std::vector<float>();
    for (ll i = 0; i < n; i++) {
        tempVector = kernelImage[i];
        for (int j = 0; j < numPrivates; j++) {
            tempVector.push_back(kernelPrivate[i][j]);
        }
        superKernel.push_back(tempVector);
    }
    ll newElem;
    for (int i = 0; i<numPrivates; i++) {
        newElem = i + n;
        indexCorrectedP.insert(newElem);
    }
    // std::cout << "Superkernel created, instantiating logDet\n";
    facLoc = new FacilityLocation(n+numPrivates, superKernel, false, std::unordered_set<ll>(), true);
    // std::cout << "Instantiated logDet instantiating condGain\n";
    condGain = new ConditionalGain(*facLoc, indexCorrectedP);
    // std::cout << "Instantiated condGain\n";
}

// FacilityLocationConditionalGain* FacilityLocationConditionalGain::clone() {
//     return NULL;
// }

double FacilityLocationConditionalGain::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "FacilityLocationConditionalGain's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    result = condGain->evaluate(X);

    return result;
}

double FacilityLocationConditionalGain::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    result = condGain->evaluateWithMemoization(X);
    return result;
}

double FacilityLocationConditionalGain::marginalGain(std::unordered_set<ll> const &X, ll item) {
    double gain = 0;

    if (X.find(item)!=X.end()) {
        return 0;
    }

    gain = condGain->marginalGain(X, item);
    return gain;
}

double FacilityLocationConditionalGain::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }
    // std::cout << "Calling condGain's marginalGainWithMemoization\n";
    gain = condGain->marginalGainWithMemoization(X, item);
    
    return gain;
}

void FacilityLocationConditionalGain::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    condGain->updateMemoization(X, item);
}

// std::vector<std::pair<ll, double>> FacilityLocationConditionalGain::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "FacilityLocationConditionalGain maximize\n";
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

std::unordered_set<ll> FacilityLocationConditionalGain::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void FacilityLocationConditionalGain::clearMemoization()
{
    condGain->clearMemoization();
}

void FacilityLocationConditionalGain::setMemoization(std::unordered_set<ll> const &X) 
{
    condGain->setMemoization(X);
}
