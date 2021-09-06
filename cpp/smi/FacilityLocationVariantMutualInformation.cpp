#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"FacilityLocationVariantMutualInformation.h"

FacilityLocationVariantMutualInformation::FacilityLocationVariantMutualInformation(ll n_, int numQueries_, std::vector<std::vector<float>> const &kernelQuery_, float magnificationLambda_): n(n_), numQueries(numQueries_), kernelQuery(kernelQuery_), magnificationLambda(magnificationLambda_){
    qMaxMod.clear();
    for (ll i = 0; i < n; i++) {
        float max = std::numeric_limits<float>::min();
        for(int q = 0; q < numQueries; q++) {
            if (kernelQuery[i][q] > max) {
                max = kernelQuery[i][q];
            }
        }
        qMaxMod.push_back(max);
    }
    similarityWithNearestInX = std::vector<float>(numQueries, 0);
}

// FacilityLocationVariantMutualInformation* FacilityLocationVariantMutualInformation::clone() {
//     return NULL;
// }

double FacilityLocationVariantMutualInformation::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "FacilityLocationVariantMutualInformation's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    for (int i = 0; i < numQueries; i++) {
        float maxcurr = std::numeric_limits<float>::min();
        for(auto elem: X) {
            if (kernelQuery[elem][i] > maxcurr) {
                maxcurr = kernelQuery[elem][i];
            }
        }
        result += maxcurr;
    }
    double sum = 0;
    for (auto elem: X) {
        sum += qMaxMod[elem];
    }
    result += magnificationLambda*sum;
    return result;
}

double FacilityLocationVariantMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    for (int i = 0; i < numQueries; i++) {
        result += similarityWithNearestInX[i];
    }
    double sum = 0;
    for (auto elem: X) {
        sum += qMaxMod[elem];
    }
    result += magnificationLambda*sum;
    return result;
}

double FacilityLocationVariantMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
        return 0;
    }
    double evalOld = evaluate(X);
    std::unordered_set<ll> newSet = X;
    newSet.insert(item);
    double evalnew = evaluate(newSet);
    return evalnew - evalOld;
}

double FacilityLocationVariantMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }

    for (int i=0; i<numQueries; i++) {
        gain += std::max(similarityWithNearestInX[i],
                     kernelQuery[item][i]) -
                 similarityWithNearestInX[i];
    }
    gain += (magnificationLambda * qMaxMod[item]);
    return gain;
}

void FacilityLocationVariantMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    for (int i = 0; i < numQueries; i++) {
        similarityWithNearestInX[i] = std::max(similarityWithNearestInX[i],
                kernelQuery[item][i]);
    }
}

// std::vector<std::pair<ll, double>> FacilityLocationVariantMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "FacilityLocationVariantMutualInformation maximize\n";
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

std::unordered_set<ll> FacilityLocationVariantMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void FacilityLocationVariantMutualInformation::clearMemoization()
{
    for (int i = 0; i < numQueries; i++) {
        similarityWithNearestInX[i] = 0;
    }
}

void FacilityLocationVariantMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
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
