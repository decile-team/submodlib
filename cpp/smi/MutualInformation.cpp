#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"MutualInformation.h"

MutualInformation::MutualInformation(SetFunction& f_, std::unordered_set<ll> querySet_): f(f_), querySet(querySet_){
    // std::cout << "Instantiating mutualInfo with querySet={";
    // for(auto temp: querySet) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "}\n";
    val_fQ = f.evaluate(querySet);
    // std::cout << "val_fQ = " << val_fQ << "\n";
    fAUQ = f.clone();
    unionPreComputeSet = querySet;
    fAUQ->setMemoization(querySet);
}

MutualInformation::~MutualInformation() {

    delete fAUQ;
}

// MutualInformation* MutualInformation::clone() {
//     return NULL;
// }

double MutualInformation::evaluate(std::unordered_set<ll> const &X) {
    if (X.size() == 0) {
        return 0;
    }
    double f_A = f.evaluate(X);
    std::unordered_set<ll> unionSet;
	unionSet = set_union(querySet, X);
	double f_AUQ = fAUQ->evaluate(unionSet);
	return f_A - f_AUQ + val_fQ;
}

double MutualInformation::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    if (X.size() == 0) {
        return 0;
    }
    double f_A = f.evaluateWithMemoization(X);
	double f_AUQ = fAUQ->evaluateWithMemoization(unionPreComputeSet);
	return f_A - f_AUQ + val_fQ;
}

double MutualInformation::marginalGain(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
        return 0;
    }
	double fGain = f.marginalGain(X, item);

	double fAUQGain;
    if (querySet.find(item) != querySet.end()) {
        fAUQGain = 0;
    } else {
        std::unordered_set<ll> unionSet;
		unionSet = set_union(querySet, X);
		fAUQGain = fAUQ->marginalGain(unionSet, item);
    }
	return fGain - fAUQGain;
}

double MutualInformation::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }
    // std::cout << "Calling logDet's marginalGainWithMemoization\n";
    double fGain = f.marginalGainWithMemoization(X, item);
    // std::cout << "fGain = " << fGain << "\n";
    double fAUQGain;
    if (querySet.find(item) != querySet.end()) {
        fAUQGain = 0;
    } else {
        // std::cout << "Calling logDetU's marginalGainWithMemoization\n";
		fAUQGain = fAUQ->marginalGainWithMemoization(unionPreComputeSet, item);
    }
    // std::cout << "fAUQGain = " << fAUQGain << "\n";
	return fGain - fAUQGain;
}

void MutualInformation::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    f.updateMemoization(X, item);
    if (querySet.find(item) != querySet.end()) {
        return;
    } else {
        fAUQ->updateMemoization(unionPreComputeSet, item);
        unionPreComputeSet.insert(item);
    }
}

 std::vector<std::pair<ll, double>> MutualInformation::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
 	// std::cout << "MutualInformation maximize\n";
 	if(optimizer == "NaiveGreedy") {
 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose);
 	} else if(optimizer == "LazyGreedy") {
         return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose);
 	} else if(optimizer == "StochasticGreedy") {
         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose);
 	} else if(optimizer == "LazierThanLazyGreedy") {
         return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose);
 	} else {
 		std::cerr << "Invalid Optimizer" << std::endl;
 	}
 }

std::unordered_set<ll> MutualInformation::getEffectiveGroundSet() {
 	std::unordered_set<ll> effectiveGroundSet;
 	effectiveGroundSet.reserve(n);
 	for (ll i = 0; i < n; ++i){
 		effectiveGroundSet.insert(i); 
 	}
 	return effectiveGroundSet;
}

void MutualInformation::clearMemoization()
{
    f.clearMemoization();
    unionPreComputeSet.clear();
    unionPreComputeSet = querySet;
    fAUQ->setMemoization(querySet); 
}

void MutualInformation::setMemoization(std::unordered_set<ll> const &X) 
{
    f.setMemoization(X);
    unionPreComputeSet.clear();
    unionPreComputeSet = set_union(querySet, X);
    fAUQ->setMemoization(unionPreComputeSet);
}
