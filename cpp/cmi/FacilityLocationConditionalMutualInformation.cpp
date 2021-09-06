#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include "FacilityLocationConditionalMutualInformation.h"

FacilityLocationConditionalMutualInformation::FacilityLocationConditionalMutualInformation(ll n_, int numQueries_, int numPrivates_, std::vector<std::vector<float>> const &kernelImage_, std::vector<std::vector<float>> const &kernelQuery_, std::vector<std::vector<float>> const &kernelPrivate_, float magnificationLambda_, float privacyHardness_): n(n_), numQueries(numQueries_), numPrivates(numPrivates_), kernelImage(kernelImage_), kernelQuery(kernelQuery_), kernelPrivate(kernelPrivate_), magnificationLambda(magnificationLambda_), privacyHardness(privacyHardness_){
    if(magnificationLambda != 1) {
        for(ll i=0; i<n; i++) {
            for(int j=0; j<numQueries; j++) {
                kernelQuery[i][j] *= magnificationLambda;
            }
        }
    }
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
        for(int j = 0; j < numQueries; j++) {
			tempVector.push_back(kernelQuery[i][j]);
		}
        for (int j = 0; j < numPrivates; j++) {
            tempVector.push_back(kernelPrivate[i][j]);
        }
        superKernel.push_back(tempVector);
    }
    ll newElem;
    for (int i = 0; i<numQueries; i++) {
        newElem = i + n;
        indexCorrectedQ.insert(newElem);
    }
    for (int i = 0; i<numPrivates; i++) {
        newElem = i + n + numQueries;
        indexCorrectedP.insert(newElem);
    }
    // std::cout << "Superkernel created, instantiating logDet\n";
    facLoc = new FacilityLocation(n+numQueries+numPrivates, superKernel, false, std::unordered_set<ll>(), true);
    // std::cout << "Instantiated logDet instantiating condGain\n";
    condGain = new ConditionalGain(*facLoc, indexCorrectedP);
    mutualInfo = new MutualInformation(*condGain, indexCorrectedQ);
    // std::cout << "Instantiated condGain\n";
}

double FacilityLocationConditionalMutualInformation::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "FacilityLocationConditionalMutualInformation's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    result = mutualInfo->evaluate(X);

    return result;
}

double FacilityLocationConditionalMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    result = mutualInfo->evaluateWithMemoization(X);
    return result;
}

double FacilityLocationConditionalMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
    double gain = 0;

    if (X.find(item)!=X.end()) {
        return 0;
    }

    gain = mutualInfo->marginalGain(X, item);
    return gain;
}

double FacilityLocationConditionalMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }
    // std::cout << "Calling mutualInfo's marginalGainWithMemoization\n";
    gain = mutualInfo->marginalGainWithMemoization(X, item);
    
    return gain;
}

void FacilityLocationConditionalMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    mutualInfo->updateMemoization(X, item);
}

// std::vector<std::pair<ll, double>> FacilityLocationConditionalMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "FacilityLocationConditionalMutualInformation maximize\n";
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

std::unordered_set<ll> FacilityLocationConditionalMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void FacilityLocationConditionalMutualInformation::clearMemoization()
{
    mutualInfo->clearMemoization();
}

void FacilityLocationConditionalMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    mutualInfo->setMemoization(X);
}
