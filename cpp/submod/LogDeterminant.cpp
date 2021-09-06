// Implementation of Algorithm 1 in
// https://proceedings.neurips.cc/paper/2018/file/dbbf603ff0e99629dda5d75b6f75f966-Paper.pdf

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../utils/helper.h"
#include "LogDeterminant.h"

double dotProduct(const std::vector<double> &x, const std::vector<double> &y) {
    double result = 0;
    for (int i = 0; i < x.size(); i++) {
        result += x[i] * y[i];
    }
    return result;
}

LogDeterminant::LogDeterminant() {}

// Constructor for dense mode
LogDeterminant::LogDeterminant(
    ll n_, std::vector<std::vector<float>> const &denseKernel_, bool partial_,
    std::unordered_set<ll> const &ground_, double lambda_)
    : n(n_),
      mode(dense),
      denseKernel(denseKernel_),
      partial(partial_),
      lambda(lambda_) {
    if (partial == true) {
        // ground set will now be the subset provided
        effectiveGroundSet = ground_;
    } else {
        // create groundSet with items 0 to n-1
        effectiveGroundSet.reserve(n);
        for (ll i = 0; i < n; ++i) {
            effectiveGroundSet.insert(i);  // each insert takes O(1) time
        }
    }
    numEffectiveGroundset = effectiveGroundSet.size();
    memoizedC = std::vector<std::vector<double>>(numEffectiveGroundset, std::vector<double>());
    prevDetVal = 0;
    memoizedD.clear();
    prevItem = -1;
    if (partial == true) {
        ll ind = 0;
        for (auto it : effectiveGroundSet) {
            //std::cout << it << "-->" << ind << "\n";
            originalToPartialIndexMap[it] = ind;
            ind += 1;
            memoizedD.push_back(sqrt(denseKernel[it][it] + lambda));
        }
    } else {
        for (ll i = 0; i < n; i++) {
            memoizedD.push_back(sqrt(denseKernel[i][i] + lambda));
        }
    }
    // std::cout << "LogDet Constructor: MemoizedD: [";
    // for(auto temp: memoizedD) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "]\n";
}

// Constructor for sparse mode
LogDeterminant::LogDeterminant(ll n_, std::vector<float> const &arr_val,
                               std::vector<ll> const &arr_count,
                               std::vector<ll> const &arr_col, double lambda_)
    : n(n_), mode(sparse), partial(false), lambda(lambda_) {
    if (arr_val.size() == 0 || arr_count.size() == 0 || arr_col.size() == 0) {
        throw "Error: Empty/Corrupt sparse similarity kernel";
    }
    sparseKernel = SparseSim(arr_val, arr_count, arr_col);
    effectiveGroundSet.reserve(n);
    for (ll i = 0; i < n; ++i) {
        effectiveGroundSet.insert(i);  // each insert takes O(1) time
    }
    numEffectiveGroundset = effectiveGroundSet.size();
    memoizedC = std::vector<std::vector<double>>(n, std::vector<double>());
    memoizedD.clear();
    prevDetVal = 0;
    for (ll i = 0; i < n; i++) {
        memoizedD.push_back(sqrt(sparseKernel.get_val(i, i) + lambda));
    }
    prevItem = -1;
    // std::cout << "MemoizedD: [";
    // for(auto temp: memoizedD) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "]\n";
    // std::cout << "Previous detVal = " << prevDetVal << "\n";
}

LogDeterminant::LogDeterminant(const LogDeterminant& f)
    : n(f.n),
      mode(f.mode),
      denseKernel(f.denseKernel),
      sparseKernel(f.sparseKernel),
      partial(f.partial),
      lambda(f.lambda), effectiveGroundSet(f.effectiveGroundSet), numEffectiveGroundset(f.numEffectiveGroundset), memoizedC(f.memoizedC), memoizedD(f.memoizedD), prevDetVal(f.prevDetVal), prevItem(f.prevItem), originalToPartialIndexMap(f.originalToPartialIndexMap) {
}

LogDeterminant* LogDeterminant::clone() {
    return new LogDeterminant(*this);
}

double LogDeterminant::evaluate(std::unordered_set<ll> const &X) {
    // since computeDeterminant doesnt work for bigger ground sets, calculating
    // eval through evalFast however, since eval should not affect the current
    // state of preCompute, restoring it to the value before
    // std::cout << "LogDet eval called\n";
    std::vector<std::vector<double>> currMemoizedC = memoizedC;
    std::vector<double> currMemoizedD = memoizedD;
    int currprevItem = prevItem;
    double currprevDetVal = prevDetVal;
    setMemoization(X);
    // std::cout << "Memoization set\n";
    // std::cout << "MemoizedD: [";
    // for(auto temp: memoizedD) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "]\n";
    double result = evaluateWithMemoization(X);
    // restore Memoized to what they were
    memoizedC = currMemoizedC;
    memoizedD = currMemoizedD;
    prevItem = currprevItem;
    prevDetVal = currprevDetVal;
    return result;
}

double LogDeterminant::evaluateWithMemoization(
    std::unordered_set<ll> const &X) {
    return prevDetVal;
}

double LogDeterminant::marginalGain(std::unordered_set<ll> const &X, ll item) {
    std::vector<std::vector<double>> currMemoizedC = memoizedC;
    std::vector<double> currMemoizedD = memoizedD;
    int currprevItem = prevItem;
    double currprevDetVal = prevDetVal;
    setMemoization(X);
    double result = marginalGainWithMemoization(X, item);
    // restore Memoized to what they were
    memoizedC = currMemoizedC;
    memoizedD = currMemoizedD;
    prevItem = currprevItem;
    prevDetVal = currprevDetVal;
    return result;
}

double LogDeterminant::marginalGainWithMemoization(
    std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    // std::cout << "LogDet's marginalGainWithMemoization called with X={";
    // for(auto temp: X) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "} and item=" << item << "\n";
    //this assumes that prevItem was the previous best, for example, when called in context of maximization
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

    if (enableChecks && effectiveX.find(item) != effectiveX.end()) {
        return 0;
    }
    if (partial && effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return 0;
    }
    ll itemIndex = (partial)?originalToPartialIndexMap[item]:item;

    if (mode == dense) {
        if (effectiveX.size() == 0) {
            gain = log(memoizedD[itemIndex] * memoizedD[itemIndex]);
            // std::cout << "Gain on empty set = " << gain << "\n";
        } else if (effectiveX.size() == 1) {
            ll prevItemIndex = (partial)?originalToPartialIndexMap[prevItem]:prevItem;
            double e = denseKernel[prevItem][item] / memoizedD[prevItemIndex];
            gain = log(memoizedD[itemIndex] * memoizedD[itemIndex] - e * e);
            // std::cout << "Gain on set of size 1 = " << gain << "\n";
        } else {
            ll prevItemIndex = (partial)?originalToPartialIndexMap[prevItem]:prevItem;
            // std::cout << "prevItemIndex = " << prevItemIndex << "\n";
            double e = (denseKernel[prevItem][item] -
                        dotProduct(memoizedC[prevItemIndex], memoizedC[itemIndex])) /
                       memoizedD[prevItemIndex];
            // std::cout << "e = " << e << "\n";
            // std::cout << "memoizedD[itemIndex] * memoizedD[itemIndex] = " << memoizedD[itemIndex] * memoizedD[itemIndex] << "\n";
            gain = log(memoizedD[itemIndex] * memoizedD[itemIndex] - e * e);
            // std::cout << "gain (3rd condition) = " << gain << "\n";
        }
    } else if (mode == sparse) {
        if (effectiveX.size() == 0) {
            gain = log(memoizedD[itemIndex] * memoizedD[itemIndex]);
        } else if (effectiveX.size() == 1) {
            ll prevItemIndex = (partial)?originalToPartialIndexMap[prevItem]:prevItem;
            double e =
                sparseKernel.get_val(prevItem, item) / memoizedD[prevItemIndex];
            gain = log(memoizedD[itemIndex] * memoizedD[itemIndex] - e * e);
        } else {
            ll prevItemIndex = (partial)?originalToPartialIndexMap[prevItem]:prevItem;
            // std::cout << "prevItemIndex = " << prevItemIndex << "; ";
            // std::cout << "sparseKernel.get_val(prevItem, item) = " << sparseKernel.get_val(prevItem, item) << "; ";
            // std::cout << "memoizedD[prevItemIndex] = " << memoizedD[prevItemIndex] <<"; ";
            double e = (sparseKernel.get_val(prevItem, item) -
                        dotProduct(memoizedC[prevItemIndex], memoizedC[itemIndex])) /
                       memoizedD[prevItemIndex];
            // std::cout << "memoizedD[itemIndex] = " << memoizedD[itemIndex] << "; ";
            // std::cout << "e = " << e <<"\n";
            gain = log(memoizedD[itemIndex] * memoizedD[itemIndex] - e * e);
        }
    } else {
        throw "Error: Only dense and sparse mode supported";
    }

    return gain;
}

void LogDeterminant::updateMemoization(std::unordered_set<ll> const &X,
                                       ll item) {
    //TODO: this assumes that prevItem was the previous best, for example when invoked in context of maximization
    // std::cout << "LogDeterminant updateMemoization\n";
    std::unordered_set<ll> effectiveX;

    if (partial == true) {
        // effectiveX = intersect(X, effectiveGroundSet)
        // std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(),
        // effectiveGroundSet.end(), std::inserter(effectiveX,
        // effectiveX.begin()));
        effectiveX = set_intersection(X, effectiveGroundSet);
    } else {
        effectiveX = X;
    }
    if (effectiveX.find(item) != effectiveX.end()) {
        return;
    }
    if (effectiveGroundSet.find(item)==effectiveGroundSet.end()) {
        return;
    }

    prevDetVal += marginalGainWithMemoization(X, item);
    //std::cout << "New previous detVal = " << prevDetVal << "\n";

    if (effectiveX.size() == 0) {
        // nothing needs to be done
        // memoizedD is already set appropriately
    } else {
        ll prevItemIndex = (partial)?originalToPartialIndexMap[prevItem]:prevItem;
        double prevDValue = memoizedD[prevItemIndex];
        // std::cout << "prevItemIndex = " << prevItemIndex << "; ";
        // std::cout << "prevDValue = " << prevDValue << "; ";
        if (mode == dense) {
            for(auto i: effectiveGroundSet) {
                ll iIndex = (partial)?originalToPartialIndexMap[i]:i;
                if (effectiveX.find(i) != effectiveX.end()) {
                    // std::cout << "Skipping i of " << i << " as it is already
                    // in sset" << std::endl;
                    continue;
                }
                double e = 0;
                if (effectiveX.size() == 1) {
                    // e = kernel[prevItem][i] / prevDValue;
                    e = denseKernel[prevItem][i] / memoizedD[prevItemIndex];
                    // vector<double> currvecC = vector<double>();
                    // currvecC.push_back(e);
                    // preComputeC.push_back(currvecC);
                    memoizedC[iIndex].push_back(e);
                } else {
                    // e = (denseKernel[prevItem][i] -
                    //      datk::innerProduct(memoizedC[prevItem],
                    //                         memoizedC[i])) /
                    //     prevDValue;
                    e = (denseKernel[prevItem][i] -
                         dotProduct(memoizedC[prevItemIndex], memoizedC[iIndex])) /
                        memoizedD[prevItemIndex];
                    memoizedC[iIndex].push_back(e);
                }
                // std::cout << "e = " << e << "\n";
                // std::cout << "memoizedD[iIndex] * memoizedD[iIndex] = " << memoizedD[iIndex] * memoizedD[iIndex] << "\n";
                memoizedD[iIndex] = sqrt(memoizedD[iIndex] * memoizedD[iIndex] - e * e);
            }

        } else if (mode == sparse) {
            //std::cout << "Updating memoizedC and memoizedD\n";
            for (ll i = 0; i < n; i++) {
                //std::cout << i << ": ";
                if (effectiveX.find(i) != effectiveX.end()) {
                    //std::cout << "skipping\n";
                    continue;
                }
                double e = 0;
                if (effectiveX.size() == 1) {
                    // e = kernel[prevItem][i] / prevDValue;
                    e = sparseKernel.get_val(prevItem, i) / memoizedD[prevItem];
                    // vector<double> currvecC = vector<double>();
                    // currvecC.push_back(e);
                    // preComputeC.push_back(currvecC);
                    memoizedC[i].push_back(e);
                } else {
                    // e = (sparseKernel.get_val(prevItem, i) -
                    //      datk::innerProduct(memoizedC[prevItem],
                    //                         memoizedC[i])) /
                    //     prevDValue;
                    e = (sparseKernel.get_val(prevItem, i) -
                         dotProduct(memoizedC[prevItem], memoizedC[i])) /
                        memoizedD[prevItem];
                    memoizedC[i].push_back(e);
                }
                memoizedD[i] = sqrt(memoizedD[i] * memoizedD[i] - e * e);
                //std::cout << "e = " << e << " memoizedD[i] = " << memoizedD[i] << "\n";
                // if(std::isnan(memoizedD[i])) {
                //     std::cout << "memoizedD[i] has become NAN!!";
                //     exit(1);
                // }
            }

        } else {
            throw "Error: Only dense and sparse mode supported";
        }
    }
    prevItem = item;
}

std::unordered_set<ll> LogDeterminant::getEffectiveGroundSet() {
    return effectiveGroundSet;
}

// std::vector<std::pair<ll, double>> LogDeterminant::maximize(
//     std::string optimizer, ll budget, bool stopIfZeroGain = false,
//     bool stopIfNegativeGain = false, float epsilon = 0.1,
//     bool verbose = false, bool showProgress=true) {
//     // std::cout << "LogDeterminant maximize\n";
//     if (optimizer == "NaiveGreedy") {
//         return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain,
//                                                stopIfNegativeGain, verbose, showProgress);
//     } else if (optimizer == "LazyGreedy") {
//         return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain,
//                                               stopIfNegativeGain, verbose, showProgress);
//     } else if (optimizer == "StochasticGreedy") {
//         return StochasticGreedyOptimizer().maximize(
//             *this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon,
//             verbose, showProgress);
//     } else if (optimizer == "LazierThanLazyGreedy") {
//         return LazierThanLazyGreedyOptimizer().maximize(
//             *this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
//     } else {
//         throw "Invalid optimizer";
//     }
// }

void LogDeterminant::cluster_init(
    ll n_, std::vector<std::vector<float>> const &denseKernel_,
    std::unordered_set<ll> const &ground_, bool partial, float lambda) {
    // std::cout << "LogDeterminant clusterInit\n";
    *this = LogDeterminant(n_, denseKernel_, partial, ground_, lambda);
}

void LogDeterminant::clearMemoization() {
    // std::cout << "LogDet clearMemoization()\n";
    memoizedC.clear();
    memoizedC = std::vector<std::vector<double>>(numEffectiveGroundset, std::vector<double>());
    prevDetVal = 0;
    prevItem = -1;
    if (mode == dense) {
        if (partial == true) {
            for (auto it : effectiveGroundSet) {
                ll index = originalToPartialIndexMap[it];
                memoizedD[index] = sqrt(denseKernel[it][it] + lambda);
            }
        } else {
            for (ll i = 0; i < n; i++) {
                memoizedD[i] = sqrt(denseKernel[i][i] + lambda);
            }
        }
    } else if (mode == sparse) {
        for (ll i = 0; i < n; i++) {
            memoizedD[i] = sqrt(sparseKernel.get_val(i, i) + lambda);
        }
    } else {
        throw "Error: Only dense and sparse mode supported";
    }
    //std::cout << "After clear memoization: \n";
    // std::cout << "MemoizedC: [";
    // for(auto temp: memoizedC) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "]\n";
    // std::cout << "MemoizedD: [";
    // for(auto temp: memoizedD) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "]\n";
    // std::cout << "Previous detVal = " << prevDetVal << "\n";
}

void LogDeterminant::setMemoization(std::unordered_set<ll> const &X) {

    //TODO: this implementation is probably wrong. I guess for updateMemoization to work correctly, the items should be added in context of greedy maximization and not arbitrarily
    // std::cout << "LogDeterminant setMemoization\n";
    clearMemoization();
    std::unordered_set<ll> temp;
    // ll best_id;
	// double best_val;
    // for(int count = 0; count < X.size(); count++) {
    //     best_id = -1;
    //     best_val = -1 * std::numeric_limits<double>::max();
    //     for (auto i: X) {
    //         if (temp.find(i) != temp.end()) { 
    //             //if this datapoint has already been included in temp, skip it
    //             continue;
    //         }
    //         double gain = marginalGainWithMemoization(temp, i);
    //         if (std::isnan(gain)) {
    //             throw "Gain turned out to be NAN";
    //         }
    //         if (gain > best_val) {
    //             best_id = i;
    //             best_val = gain;
    //         }
    //     }
    //     updateMemoization(temp, best_id);
    //     temp.insert(best_id);
    // }
    
    for (auto elem : X) {
        // std::cout << "Memoization for adding " << elem << " to {";
        // for (auto tmp: temp) {
        //     std::cout << tmp <<", ";
        // }
        // std::cout << "}\n";
        updateMemoization(temp, elem);
        temp.insert(elem);
    }
}
