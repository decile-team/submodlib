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

DisparitySum::DisparitySum(){}

// Constructor for dense mode
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
    if(partial == true) {
		ll ind = 0;
		for (auto it: effectiveGroundSet) {
			originalToPartialIndexMap[it] = ind;
			ind += 1;
		}
	}
}

// DisparitySum* DisparitySum::clone() {
//     return NULL;
// }

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
double get_sum_dense(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj) {
    double sum = 0;
    // for (auto it = dataset_ind.begin(); it != dataset_ind.end(); ++it) {
    //     for (auto nextIt = std::next(it, 1); nextIt != dataset_ind.end(); ++nextIt) {
    //         sum += (1 - obj.denseKernel[*it][*nextIt]);
    //     }
    // }
    //std::cout << "Inside get_sum_dense\n";
    for(auto elem1: dataset_ind) {
        for(auto elem2: dataset_ind) {
            //std::cout << "Adding 1-" << obj.denseKernel[elem1][elem2] << "\n";
            sum += (1 - obj.denseKernel[elem1][elem2]);
        }
    }
    return sum/2;
}

double get_sum_sparse(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj) {
    double sum = 0;
    // for (auto it = dataset_ind.begin(); it != dataset_ind.end(); ++it) {
    //     for (auto nextIt = std::next(it, 1); nextIt != dataset_ind.end(); ++nextIt) {
    //         sum += (1 - obj.sparseKernel.get_val(*it, *nextIt));
    //     }
    // }
    for(auto elem1: dataset_ind) {
        for(auto elem2: dataset_ind) {
            //std::cout << elem1 << ", " << elem2 << ": " << obj.sparseKernel.get_val(elem1, elem2) << "\n";
            sum += (1 - obj.sparseKernel.get_val(elem1, elem2));
        }
    }
    return sum/2;
}

double DisparitySum::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "DisparitySum's evaluate called\n";
    std::unordered_set<ll> effectiveX;
    double result = 0;

    // std::cout << "X = { ";
    // for(auto elem: X) {
    //     std::cout << elem << ", ";
    // }
    // std::cout << "}\n";

    if (partial == true) {
        //std::cout << "Partial is TRUE!!!!\n";
        // effectiveX = intersect(X, effectiveGroundSet)
        // std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(),
        //                       effectiveGroundSet.end(),
        //                       std::inserter(effectiveX, effectiveX.begin()));
        effectiveX = set_intersection(X, effectiveGroundSet);
    } else {
        effectiveX = X;
    }

    // std::cout << "EffectiveX = { ";
    // for(auto elem: effectiveX) {
    //     std::cout << elem << ", ";
    // }
    // std::cout << "}\n";

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

double DisparitySum::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    return currentSum;
}

double DisparitySum::marginalGain(std::unordered_set<ll> const &X, ll item) {
    std::unordered_set<ll> effectiveX;
    double gain = 0;

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

    if (effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return 0;
    }

    if(mode == dense) {
        for (auto elem: effectiveX) {
            gain += (1 - denseKernel[elem][item]);
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

double DisparitySum::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    //identical to marginalGain, but duplicating here to save an extra function call
    std::unordered_set<ll> effectiveX;
    double gain = 0;

    if (partial == true) {
        // effectiveX = intersect(X, effectiveGroundSet)
        // std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(),
        //                       effectiveGroundSet.end(),
        //                       std::inserter(effectiveX, effectiveX.begin()));
        effectiveX = set_intersection(X, effectiveGroundSet);
    } else {
        effectiveX = X;
    }

    if (enableChecks && effectiveX.find(item)!=effectiveX.end()) {
        return 0;
    }

    if (partial && effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return 0;
    }

    if(mode == dense) {
        for (auto elem: effectiveX) {
            gain += (1 - denseKernel[elem][item]);
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

void DisparitySum::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    currentSum += marginalGain(X, item);
}

std::unordered_set<ll> DisparitySum::getEffectiveGroundSet() {
    return effectiveGroundSet;
}

// std::vector<std::pair<ll, double>> DisparitySum::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "DisparitySum maximize\n";
// 	if(optimizer == "NaiveGreedy") {
// 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "LazyGreedy") {
//         throw "Being non submodular, DisparitySum doesn't support LazyGreedy maximization";
// 	} else if(optimizer == "StochasticGreedy") {
//         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else if(optimizer == "LazierThanLazyGreedy") {
//         throw "Being non submodular, DisparitySum doesn't support LazierThanLazyGreedy maximization";
// 	} else {
// 		throw "Invalid optimizer";
// 	}
// }

void DisparitySum::cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda) {
	// std::cout << "DisparitySum clusterInit\n";
	*this = DisparitySum(n_, denseKernel_, partial, ground_);
}

void DisparitySum::clearMemoization()
{
    currentSum=0;	
}

void DisparitySum::setMemoization(std::unordered_set<ll> const &X) 
{
    currentSum=evaluate(X);
}
