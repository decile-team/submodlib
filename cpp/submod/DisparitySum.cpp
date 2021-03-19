#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"DisparitySum.h"

DisparitySum::DisparitySum() {}

// COnstructor for dense mode
DisparitySum::DisparitySum(ll n_, std::vector<std::vector<float>> const &denseKernel_, bool partial_, std::unordered_set<ll> const &ground_): n(n_), mode(dense), denseKernel(denseKernel_), partial(partial_) {
    if (partial == true) {
        //ground set will now be the subset provided
        effectiveGroundSet = ground_;
    } else {
        //create groundSet with items 0 to n-1
		effectiveGroundSet.reserve(n);
		for (ll i = 0; i < n; ++i){
			effectiveGroundSet.insert(i); //each insert takes O(1) time
		}
    }
    numEffectiveGroundset = effectiveGroundSet.size();
    currentSum = 0;
}

// Constructor for sparse mode
DisparitySum::DisparitySum(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col): n(n_), mode(sparse), partial(false) {
    if (arr_val.size() == 0 || arr_count.size() == 0 || arr_col.size() == 0) {
		throw "Error: Empty/Corrupt sparse similarity kernel";
	}
    sparseKernel = SparseSim(arr_val, arr_count, arr_col);
    effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i) {
		effectiveGroundSet.insert(i); //each insert takes O(1) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();
    currentSum = 0;
}

// helper friend function
float get_sum_dense(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj) {
    float sum = 0;
    for (auto it = dataset_ind.begin(); it != dataset_ind.end(); ++it) {
        for (auto nextIt = std::next(it, 1); nextIt != dataset_ind.end(); ++nextIt) {
            sum += (1 - obj.denseKernel[*it][*nextIt]);
        }
    }
    return sum;
}

float get_sum_sparse(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj) {
    float sum = 0;
    for (auto it = dataset_ind.begin(); it != dataset_ind.end(); ++it) {
        for (auto nextIt = std::next(it, 1); nextIt != dataset_ind.end(); ++nextIt) {
            sum += (1 - obj.sparseKernel.get_val(*it, *nextIt));
        }
    }
    return sum;
}

float DisparitySum::evaluate(std::unordered_set<ll> const &X) {
    std::unordered_set<ll> effectiveX;
    float result = 0;

    if (partial == true) {
        // effectiveX = intersect(X, effectiveGroundSet)
        // std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(),
        //                       effectiveGroundSet.end(),
        //                       std::inserter(effectiveX, effectiveX.begin()));
        effectiveX = set_intersection(X, effectiveGroundSet);
    } else {
        effectiveX = X;
    }

    if (effectiveX.size() == 0) {
        return 0;
    }

    if (mode == dense) {
        result = get_sum_dense(effectiveX, *this);
    } else if(mode ==sparse) {
        result = get_sum_sparse(effectiveX, *this);
    } else {
        throw "Error: Only dense and sparse mode supported";
    }
    return result;
}

float DisparitySum::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    std::unordered_set<ll> effectiveX;
    float result = 0;

    if (partial == true) {
        // effectiveX = intersect(X, effectiveGroundSet)
        // std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(),
        //                       effectiveGroundSet.end(),
        //                       std::inserter(effectiveX, effectiveX.begin()));
        effectiveX = set_intersection(X, effectiveGroundSet);
    } else {
        effectiveX = X;
    }
    if (effectiveX.size() == 0)  
    {
        return 0;
    }
    return currentSum;
}

float DisparitySum::marginalGain(std::unordered_set<ll> const &X, ll item) {
    std::unordered_set<ll> effectiveX;
    float gain = 0;

    if (partial == true) {
        // effectiveX = intersect(X, effectiveGroundSet)
        // std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(),
        //                       effectiveGroundSet.end(),
        //                       std::inserter(effectiveX, effectiveX.begin()));
        effectiveX = set_intersection(X, effectiveGroundSet);
    } else {
        effectiveX = X;
    }

    if (effectiveX.find(item)!=effectiveX.end()) {
        return 0;
    }

    if(mode == dense) {
        for (auto elem: effectiveX) {
            gain += (1 - denseKernel[item][elem]);
        }
    } else if (mode == sparse) {
        for (auto elem: effectiveX) {
            gain += (1 - sparseKernel.get_val(item, elem));
        }
    } else {
        throw "Error: Only dense and sparse mode supported";
    }
    
    return gain;
}

float DisparitySum::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item) {
    return marginalGain(X, item);
}

void DisparitySum::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    currentSum += marginalGain(X, item);
}

std::unordered_set<ll> DisparitySum::getEffectiveGroundSet() {
    return effectiveGroundSet;
}

std::vector<std::pair<ll, float>> DisparitySum::maximize(std::string optimizer,float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false) {
	// std::cout << "DisparitySum maximize\n";
	if(optimizer == "NaiveGreedy") {
		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose);
	} else if(optimizer == "LazyGreedy") {
        return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose);
	} else if(optimizer == "StochasticGreedy") {
        return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose);
	} else if(optimizer == "LazierThanLazyGreedy") {
        return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose);
	} else {
		std::cerr << "Optimizer not yet implemented" << std::endl;
	}
}

void DisparitySum::clearMemoization()
{
    currentSum=0;	
}

void DisparitySum::setMemoization(std::unordered_set<ll> const &X) 
{
    currentSum=evaluate(X);
}
