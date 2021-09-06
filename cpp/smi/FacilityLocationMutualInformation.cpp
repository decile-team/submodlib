#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"FacilityLocationMutualInformation.h"

FacilityLocationMutualInformation::FacilityLocationMutualInformation(ll n_, int numQueries_, std::vector<std::vector<float>> const &kernelImage_, std::vector<std::vector<float>> const &kernelQuery_, float magnificationLambda_): n(n_), numQueries(numQueries_), kernelImage(kernelImage_), kernelQuery(kernelQuery_), magnificationLambda(magnificationLambda_){
    qMaxMod.clear();
    if(magnificationLambda != 1) {
        for(ll i=0; i<n; i++) {
            for(int j=0; j<numQueries; j++) {
                kernelQuery[i][j] *= magnificationLambda;
            }
        }
    }
    for (ll i = 0; i < n; i++) {
        float max = std::numeric_limits<float>::min();
        for(int q = 0; q < numQueries; q++) {
            if (kernelQuery[i][q] > max) {
                max = kernelQuery[i][q];
            }
        }
        qMaxMod.push_back(max);
    }
    similarityWithNearestInX = std::vector<float>(n, 0);
}

// FacilityLocationMutualInformation* FacilityLocationMutualInformation::clone() {
//     return NULL;
// }

double FacilityLocationMutualInformation::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "FacilityLocationMutualInformation's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    for (ll i = 0; i < n; i++) {
        float maxcurr = std::numeric_limits<float>::min();
        for(auto elem: X) {
            if (kernelImage[i][elem] > maxcurr) {
                maxcurr = kernelImage[i][elem];
            }
        }
        result += std::min(maxcurr, qMaxMod[i]);
    }
    return result;
}

double FacilityLocationMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    for (ll i = 0; i < n; i++) {
        result += std::min(similarityWithNearestInX[i], qMaxMod[i]);
    }
    return result;
}

double FacilityLocationMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
    double gain = 0;

    if (X.find(item)!=X.end()) {
        return 0;
    }

    for (ll i = 0; i < n; i++) {
        float maxcurr = 0;
        for(auto elem: X) {
            if (kernelImage[i][elem] > maxcurr) {
                maxcurr = kernelImage[i][elem];
            }
        }
        gain += std::min(std::max(maxcurr, kernelImage[i][item]), qMaxMod[i]) -
                 std::min(maxcurr, qMaxMod[i]);
    }
    return gain;
}

double FacilityLocationMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }

    for (ll i = 0; i < n; i++) {
        gain += std::min(std::max(similarityWithNearestInX[i], kernelImage[i][item]), qMaxMod[i]) -
                 std::min(similarityWithNearestInX[i], qMaxMod[i]);
    }
    return gain;
}

void FacilityLocationMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    for (ll i = 0; i < n; i++) {
        similarityWithNearestInX[i] = std::max(similarityWithNearestInX[i], kernelImage[i][item]);
    }
}

// std::vector<std::pair<ll, double>> FacilityLocationMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "FacilityLocationMutualInformation maximize\n";
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

std::unordered_set<ll> FacilityLocationMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void FacilityLocationMutualInformation::clearMemoization()
{
    for (ll i = 0; i < n; i++) {
        similarityWithNearestInX[i] = 0;
    }
}

void FacilityLocationMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
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
