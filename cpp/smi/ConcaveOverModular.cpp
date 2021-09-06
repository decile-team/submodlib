#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"ConcaveOverModular.h"

ConcaveOverModular::ConcaveOverModular(ll n_, int numQueries_, std::vector<std::vector<float>> const &kernelQuery_, float queryDiversityEta_, Type type_): n(n_), numQueries(numQueries_), kernelQuery(kernelQuery_), queryDiversityEta(queryDiversityEta_), type(type_){
    querySumForEachImage.clear();
    for (ll i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j<numQueries; j++) {
            sum += kernelQuery[i][j];
        }
        querySumForEachImage.push_back(sum);
    }
    subsetSumForEachQuery = std::vector<double>(numQueries, 0);
}

// ConcaveOverModular* ConcaveOverModular::clone() {
//     return NULL;
// }

double ConcaveOverModular::transform(double val) {
	//std::cout << "Type = " << type << "\n";
	switch(type) {
		case inverse:
		    //std::cout << "Inverse\n";
		    return (1-1/(val+1));
		case logarithmic:
			//std::cout << "Logarithmic\n";
		    return log(1 + val);
		case squareRoot:
		    //std::cout << "Sqrt\n";
		    return sqrt(val);
	}
}

double ConcaveOverModular::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "ConcaveOverModular's evaluate called\n";
    double result = 0;

    if (X.size() == 0) {
        return 0;
    }

    double sum1 = 0;
    for (auto it: X) {
        double sum = 0;
        for (int it2=0; it2 < numQueries; it2++) {
            sum += kernelQuery[it][it2];
        }
        sum1 += transform(sum);
    }
    result += queryDiversityEta * sum1;

    double sum2 = 0;
    for (int it=0; it<numQueries; it++) {
        double sum = 0;
        for (auto it2: X) {
            sum += kernelQuery[it2][it];
        }
        sum2 += transform(sum);
    }

    result += sum2;

    return result;
}

double ConcaveOverModular::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    double result = 0;
    if (X.size() == 0) {
        return 0;
    }
    double sum1 = 0;
    for (auto it: X) {
        sum1 += transform(querySumForEachImage[it]);
    }
    result += queryDiversityEta * sum1;

    double sum2 = 0;
    for (int it=0; it < numQueries; it++) {
        sum2 += transform(subsetSumForEachQuery[it]);
    }

    result += sum2;
    return result;
}

double ConcaveOverModular::marginalGain(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
        return 0;
    }
    double gain = 0;
    double sum = 0;
    for (int it=0; it<numQueries; it++) {
        sum += kernelQuery[item][it];
    }
    gain += queryDiversityEta * transform(sum);

    sum = 0;
    for (int it=0; it<numQueries; it++) {
        double sum2 = 0;
        for (auto it2: X) {
            sum2 += kernelQuery[it2][it];
        }
        sum2 += kernelQuery[item][it];
        sum += transform(sum2);
    }
    gain += sum;

    sum = 0;
    for (int it=0; it<numQueries; it++) {
        double sum2 = 0;
        for (auto it2: X) {
            sum2 += kernelQuery[it2][it];
        }
        sum += transform(sum2);
    }
    gain -= sum;
    return gain;
}

double ConcaveOverModular::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    double gain = 0;

    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }

    gain += queryDiversityEta * transform(querySumForEachImage[item]);

    double sum = 0;
    for (int it=0; it<numQueries; it++) {
        double sum2 = subsetSumForEachQuery[it];
        sum2 += kernelQuery[item][it];
        sum += transform(sum2);
    }
    gain += sum;

    sum = 0;
    for (int it=0; it<numQueries; it++) {
        sum += transform(subsetSumForEachQuery[it]);
    }

    gain -= sum;
    return gain;
}

void ConcaveOverModular::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    for (int i = 0; i < numQueries; i++) {
        subsetSumForEachQuery[i] +=
            kernelQuery[item][i];
    }
}

// std::vector<std::pair<ll, double>> ConcaveOverModular::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
// 	// std::cout << "ConcaveOverModular maximize\n";
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

std::unordered_set<ll> ConcaveOverModular::getEffectiveGroundSet() {
	std::unordered_set<ll> effectiveGroundSet;
	effectiveGroundSet.reserve(n);
	for (ll i = 0; i < n; ++i){
		effectiveGroundSet.insert(i); 
	}
	return effectiveGroundSet;
}

void ConcaveOverModular::clearMemoization()
{
    for (int i = 0; i < numQueries; i++) {
        subsetSumForEachQuery[i] = 0;
    }   
}

void ConcaveOverModular::setMemoization(std::unordered_set<ll> const &X) 
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
