#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"LogDeterminantMutualInformation.h"

LogDeterminantMutualInformation::LogDeterminantMutualInformation(ll n_, int numQueries_, std::vector<std::vector<float>> const &kernelImage_, std::vector<std::vector<float>> const &kernelQuery_, std::vector<std::vector<float>> const &kernelQueryQuery_, double lambda_, float magnificationLambda_): n(n_), numQueries(numQueries_), kernelImage(kernelImage_), kernelQuery(kernelQuery_), kernelQueryQuery(kernelQueryQuery_), lambda(lambda_), magnificationLambda(magnificationLambda_){
    if(magnificationLambda != 1) {
        for(ll i=0; i<n; i++) {
            for(int j=0; j<numQueries; j++) {
                kernelQuery[i][j] *= magnificationLambda;
            }
        }
    }
    std::vector<float> tempVector = std::vector<float>();
    for (ll i = 0; i < n; i++) {
        tempVector = kernelImage[i];
        for (int j = 0; j < numQueries; j++) {
            tempVector.push_back(kernelQuery[i][j]);
        }
        superKernel.push_back(tempVector);
    }
    ll totalNumElem = n + numQueries;
    for (int i = 0; i < numQueries; i++) {
        std::vector<float> queryRow = std::vector<float>();
        for (ll j = 0; j < totalNumElem; j++) {
            if (j < n) {
                queryRow.push_back(kernelQuery[j][i]);
            } else {
                queryRow.push_back(kernelQueryQuery[i][j - n]);
            }
        }
        superKernel.push_back(queryRow);
    }
    ll newElem;
    for (int i = 0; i<numQueries; i++) {
        newElem = i + n;
        indexCorrectedQ.insert(newElem);
    }
    // std::cout << "Superkernel created, instantiating logDet\n";
    logDet = new LogDeterminant(totalNumElem, superKernel, false, std::unordered_set<ll>(), lambda);
    // std::cout << "Instantiated logDet instantiating mutualInfo\n";
    mutualInfo = new MutualInformation(*logDet, indexCorrectedQ);
    // std::cout << "Instantiated mutualInfo\n";
}

LogDeterminantMutualInformation::~LogDeterminantMutualInformation() {

    delete mutualInfo;
    delete logDet;
}

// LogDeterminantMutualInformation* LogDeterminantMutualInformation::clone() {
//     return NULL;
// }

double LogDeterminantMutualInformation::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "LogDeterminantMutualInformation's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    result = mutualInfo->evaluate(X);

    return result;
}

double LogDeterminantMutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    result = mutualInfo->evaluateWithMemoization(X);
    return result;
}

double LogDeterminantMutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
    double gain = 0;

    if (X.find(item)!=X.end()) {
        return 0;
    }

    gain = mutualInfo->marginalGain(X, item);
    return gain;
}

double LogDeterminantMutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }
    // std::cout << "Calling mutualInfo's marginalGainWithMemoization\n";
    gain = mutualInfo->marginalGainWithMemoization(X, item);
    
    return gain;
}

void LogDeterminantMutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    mutualInfo->updateMemoization(X, item);
}

// std::vector<std::pair<ll, double>> LogDeterminantMutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "LogDeterminantMutualInformation maximize\n";
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

std::unordered_set<ll> LogDeterminantMutualInformation::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void LogDeterminantMutualInformation::clearMemoization()
{
    mutualInfo->clearMemoization();
}

void LogDeterminantMutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    mutualInfo->setMemoization(X);
}
