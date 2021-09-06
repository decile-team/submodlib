#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"DisparityMin.h"

DisparityMin::DisparityMin(){}

// Constructor for dense mode
DisparityMin::DisparityMin(ll n_, std::vector<std::vector<float>> const &denseKernel_, bool partial_, std::unordered_set<ll> const &ground_): n(n_), mode(dense), denseKernel(denseKernel_), partial(partial_) {
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
    currentMin = 0;
    if(partial == true) {
		ll ind = 0;
		for (auto it: effectiveGroundSet) {
			originalToPartialIndexMap[it] = ind;
			ind += 1;
		}
	}
}

// DisparityMin* DisparityMin::clone() {
//     return NULL;
// }
// Constructor for sparse mode
DisparityMin::DisparityMin(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col): n(n_), mode(sparse), partial(false) {
    if (arr_val.size() == 0 || arr_count.size() == 0 || arr_col.size() == 0) {
		throw "Error: Empty/Corrupt sparse similarity kernel";
	}
    sparseKernel = SparseSim(arr_val, arr_count, arr_col);
    effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i) {
		effectiveGroundSet.insert(i); //each insert takes O(1) time
	}
	numEffectiveGroundset = effectiveGroundSet.size();
    currentMin = 0;
}

// helper friend function
double get_min_dense(std::unordered_set<ll> const &dataset_ind, DisparityMin &obj) {
    double min = 1;
    // for (auto it = dataset_ind.begin(); it != dataset_ind.end(); ++it) {
    //     for (auto nextIt = std::next(it, 1); nextIt != dataset_ind.end(); ++nextIt) {
    //         sum += (1 - obj.denseKernel[*it][*nextIt]);
    //     }
    // }
    //std::cout << "Inside get_min_dense\n";
    for(auto elem1: dataset_ind) {
        for(auto elem2: dataset_ind) {
            //std::cout << "Adding 1-" << obj.denseKernel[elem1][elem2] << "\n";
            if((1 - obj.denseKernel[elem1][elem2] < min) && elem1 != elem2) {
                min = 1 - obj.denseKernel[elem1][elem2];
            }
        }
    }
    return min;
}

double get_min_sparse(std::unordered_set<ll> const &dataset_ind, DisparityMin &obj) {
    double min = 1;
    // for (auto it = dataset_ind.begin(); it != dataset_ind.end(); ++it) {
    //     for (auto nextIt = std::next(it, 1); nextIt != dataset_ind.end(); ++nextIt) {
    //         sum += (1 - obj.sparseKernel.get_val(*it, *nextIt));
    //     }
    // }
    for(auto elem1: dataset_ind) {
        for(auto elem2: dataset_ind) {
            //std::cout << elem1 << ", " << elem2 << ": " << obj.sparseKernel.get_val(elem1, elem2) << "\n";
            if((1 - obj.sparseKernel.get_val(elem1, elem2) < min) && elem1 != elem2) {
                min = 1 - obj.sparseKernel.get_val(elem1, elem2);
            }
        }
    }
    return min;
}

double DisparityMin::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "DisparityMin's evaluate called\n";
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

    if (effectiveX.size() == 0 || effectiveX.size() == 1) {
        return 0;
    }

    if (mode == dense) {
        result = get_min_dense(effectiveX, *this);
    } else if(mode ==sparse) {
        result = get_min_sparse(effectiveX, *this);
    } else {
        throw "Error: Only dense and sparse mode supported";
    }
    return result;
}

double DisparityMin::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    return currentMin;
}

double DisparityMin::marginalGain(std::unordered_set<ll> const &X, ll item) {
    std::unordered_set<ll> effectiveX;

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

    double min;
    if(effectiveX.size() == 1) {
        min = 1;
    } else {
        min = currentMin;
    }

    if(mode == dense) {
        for (auto elem: effectiveX) {
            if((1 - denseKernel[elem][item] < min) && elem!=item) {
                min = 1 - denseKernel[elem][item];
            }
        }
    } else if (mode == sparse) {
        for (auto elem: effectiveX) {
            if((1 - sparseKernel.get_val(item, elem) < min) && elem!=item) {
                min = 1 - sparseKernel.get_val(item, elem);
            }
        }
    } else {
        throw "Error: Only dense and sparse mode supported";
    }
    
    return min-currentMin;
}

double DisparityMin::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    //identical to marginalGain, but duplicating here to save an extra function call
    std::unordered_set<ll> effectiveX;

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

    double min;
    if(effectiveX.size() == 1) {
        min = 1;
    } else {
        min = currentMin;
    }

    if(mode == dense) {
        for (auto elem: effectiveX) {
            if((1 - denseKernel[elem][item] < min) && elem!=item) {
                min = 1 - denseKernel[elem][item];
            }
        }
    } else if (mode == sparse) {
        for (auto elem: effectiveX) {
            if((1 - sparseKernel.get_val(item, elem) < min) && elem!=item) {
                min = 1 - sparseKernel.get_val(item, elem);
            }
        }
    } else {
        throw "Error: Only dense and sparse mode supported";
    }
    
    return min-currentMin;
}

void DisparityMin::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    std::unordered_set<ll> effectiveX;

    if (partial == true) {
        // effectiveX = intersect(X, effectiveGroundSet)
        // std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(),
        //                       effectiveGroundSet.end(),
        //                       std::inserter(effectiveX, effectiveX.begin()));
        effectiveX = set_intersection(X, effectiveGroundSet);
    } else {
        effectiveX = X;
    }
    if(effectiveX.find(item) != effectiveX.end()) {
        return;
    }
    if (effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return;
    }
    if(effectiveX.size() == 1) {
        if(mode == dense) {
            for(auto elem: effectiveX) {
                currentMin = 1 - denseKernel[elem][item];
            }
        } else if(mode == sparse) {
            for(auto elem: effectiveX) {
                currentMin = 1 - sparseKernel.get_val(elem, item);
            }
        } else {
            throw "Error: Only dense and sparse mode supported";
        }
    } else {
        if(mode == dense) {
            for(auto elem: effectiveX) {
                if((1 - denseKernel[elem][item] < currentMin) && elem!=item) {
                    currentMin = 1- denseKernel[elem][item];
                }
            }
        } else if(mode == sparse) {
            for(auto elem: effectiveX) {
                if((1 - sparseKernel.get_val(elem, item) < currentMin) && elem!=item) {
                    currentMin = 1- sparseKernel.get_val(elem, item);
                }
            }
        } else {
            throw "Error: Only dense and sparse mode supported";
        }
    }
}

std::unordered_set<ll> DisparityMin::getEffectiveGroundSet() {
    return effectiveGroundSet;
}

// std::vector<std::pair<ll, double>> DisparityMin::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "DisparityMin maximize\n";
// 	if(optimizer == "NaiveGreedy") {
// 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if(optimizer == "LazyGreedy") {
//         throw "Being non submodular, DisparityMin doesn't support LazyGreedy maximization";
// 	} else if(optimizer == "StochasticGreedy") {
//         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
// 	} else if(optimizer == "LazierThanLazyGreedy") {
//         throw "Being non submodular, DisparityMin doesn't support LazierThanLazyGreedy maximization";
// 	} else {
// 		throw "Invalid optimizer";
// 	}
// }

void DisparityMin::cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda) {
	// std::cout << "DisparityMin clusterInit\n";
	*this = DisparityMin(n_, denseKernel_, partial, ground_);
}

void DisparityMin::clearMemoization()
{
    currentMin=0;	
}

void DisparityMin::setMemoization(std::unordered_set<ll> const &X) 
{
    currentMin=evaluate(X);
}
