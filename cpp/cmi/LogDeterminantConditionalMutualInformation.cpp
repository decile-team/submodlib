#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"LogDeterminantConditionalMutualInformation.h"

LogDeterminantConditionalMutualInformation::LogDeterminantConditionalMutualInformation(
        ll n_, int numQueries_, int numPrivates_,
        std::vector<std::vector<float>> const &kernelImage_,
        std::vector<std::vector<float>> const &kernelQuery_,
        std::vector<std::vector<float>> const &kernelQueryQuery_,
        std::vector<std::vector<float>> const &kernelPrivate_,
        std::vector<std::vector<float>> const &kernelPrivatePrivate_,
        std::vector<std::vector<float>> const &kernelQueryPrivate_,
        double lambda_, float magnificationLambda_, float privacyHardness_): n(n_), numQueries(numQueries_), numPrivates(numPrivates_), kernelImage(kernelImage_), kernelQuery(kernelQuery_), kernelQueryQuery(kernelQueryQuery_), kernelPrivate(kernelPrivate_), kernelPrivatePrivate(kernelPrivatePrivate_), kernelQueryPrivate(kernelQueryPrivate_), lambda(lambda_), magnificationLambda(magnificationLambda_), privacyHardness(privacyHardness_){
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
        for (int j = 0; j < numQueries; j++) {
            tempVector.push_back(kernelQuery[i][j]);
        }
        for (int j = 0; j < numPrivates; j++) {
            tempVector.push_back(kernelPrivate[i][j]);
        }
        superKernel.push_back(tempVector);
    }
    ll totalNumElem = n + numQueries + numPrivates;
    for (int i = 0; i < numQueries; i++) {
        std::vector<float> queryRow = std::vector<float>();
        for (ll j = 0; j < totalNumElem; j++) {
            if (j < n) {
                queryRow.push_back(kernelQuery[j][i]);
            } else if(j>=n && j<(n+numQueries)) {
                queryRow.push_back(
                    kernelQueryQuery[i][j - n]);
            } else {
                queryRow.push_back(kernelQueryPrivate[i][j - (n+numQueries)]);
            }
        }
        superKernel.push_back(queryRow);
    }
    for (int i = 0; i < numPrivates; i++) {
        std::vector<float> privateRow = std::vector<float>();
        for (int j = 0; j < totalNumElem; j++) {
            if (j < n) {
                privateRow.push_back(kernelPrivate[j][i]);
            } else if(j>= n && j < (n + numQueries)) {
                privateRow.push_back(
                    kernelQueryPrivate[j - n][i]);
            } else {
                privateRow.push_back(
                    kernelPrivatePrivate[i][j - (n + numQueries)]);
            }
        }
        superKernel.push_back(privateRow);
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
    logDet = new LogDeterminant(totalNumElem, superKernel, false, std::unordered_set<ll>(), lambda);
    condGain = new ConditionalGain(*logDet, indexCorrectedP);
    mutualInfo = new MutualInformation(*condGain, indexCorrectedQ);
}

// LogDeterminantConditionalMutualInformation* LogDeterminantConditionalMutualInformation::clone() {
//     return NULL;
// }

double LogDeterminantConditionalMutualInformation::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "LogDeterminantConditionalMutualInformation's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    result = mutualInfo->evaluate(X);

    return result;
}

double LogDeterminantConditionalMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    result = mutualInfo->evaluateWithMemoization(X);
    return result;
}

double LogDeterminantConditionalMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
    double gain = 0;

    if (X.find(item)!=X.end()) {
        return 0;
    }

    gain = mutualInfo->marginalGain(X, item);
    return gain;
}

double LogDeterminantConditionalMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }
    // std::cout << "Calling mutualInfo's marginalGainWithMemoization\n";
    gain = mutualInfo->marginalGainWithMemoization(X, item);
    
    return gain;
}

void LogDeterminantConditionalMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    mutualInfo->updateMemoization(X, item);
}

// std::vector<std::pair<ll, double>> LogDeterminantConditionalMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "LogDeterminantConditionalMutualInformation maximize\n";
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

std::unordered_set<ll> LogDeterminantConditionalMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void LogDeterminantConditionalMutualInformation::clearMemoization()
{
    mutualInfo->clearMemoization();
}

void LogDeterminantConditionalMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    mutualInfo->setMemoization(X);
}
