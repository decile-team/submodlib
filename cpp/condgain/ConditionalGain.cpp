#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include "../utils/helper.h"
#include"ConditionalGain.h"

ConditionalGain::ConditionalGain(SetFunction& f_, std::unordered_set<ll> privateSet_): f(f_), privateSet(privateSet_){
    // std::cout << "Instantiating mutualInfo with privateSet={";
    // for(auto temp: privateSet) {
    //     std::cout << temp << ", ";
    // }
    // std::cout << "}\n";
    val_fP = f.evaluate(privateSet);
    //std::cout << "val_fP = " << val_fP << "\n";
    //fAUP = f.clone();
    unionPreComputeSet = privateSet;
    //fAUP->setMemoization(privateSet);
    f.setMemoization(privateSet);
}

ConditionalGain::ConditionalGain(const ConditionalGain& input_f): privateSet(input_f.privateSet), val_fP(input_f.val_fP), unionPreComputeSet(input_f.unionPreComputeSet), f(*(input_f.f.clone())) {
}

ConditionalGain* ConditionalGain::clone() {
    return new ConditionalGain(*this);
}

double ConditionalGain::evaluate(std::unordered_set<ll> const &X) {
    if (X.size() == 0) {
        return 0;
    }
    std::unordered_set<ll> unionSet;
	unionSet = set_union(privateSet, X);
    // std::cout << "unionSet={";
    // for(auto elem: unionSet) {
    //     std::cout << elem <<", ";
    // }
    // std::cout <<"}\n";
	//double f_AUP = fAUP->evaluate(unionSet);
    double f_AUP = f.evaluate(unionSet);
    //std::cout << "f_AUP(normal) = " << f_AUP << "\n";
	return f_AUP - val_fP;
}

double ConditionalGain::evaluateWithMemoization(std::unordered_set<ll> const &X) { 
    if (X.size() == 0) {
        return 0;
    }
	//double f_AUP = fAUP->evaluateWithMemoization(unionPreComputeSet);
    double f_AUP = f.evaluateWithMemoization(unionPreComputeSet);
    //std::cout << "f_AUP(fast) = " << f_AUP << "\n";
	return f_AUP - val_fP;
}

double ConditionalGain::marginalGain(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
        return 0;
    }
	double fAUPGain;
    if (privateSet.find(item) != privateSet.end()) {
        fAUPGain = 0;
    } else {
        std::unordered_set<ll> unionSet;
		unionSet = set_union(privateSet, X);
		//fAUPGain = fAUP->marginalGain(unionSet, item);
        fAUPGain = f.marginalGain(unionSet, item);
    }
	return fAUPGain;
}

double ConditionalGain::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    if (enableChecks && X.find(item)!=X.end()) {
        return 0;
    }
    // std::cout << "Calling logDet's marginalGainWithMemoization\n";
    // std::cout << "fGain = " << fGain << "\n";
    double fAUPGain;
    if (privateSet.find(item) != privateSet.end()) {
        fAUPGain = 0;
    } else {
        // std::cout << "Calling logDetU's marginalGainWithMemoization\n";
		//fAUPGain = fAUP->marginalGainWithMemoization(unionPreComputeSet, item);
        fAUPGain = f.marginalGainWithMemoization(unionPreComputeSet, item);
    }
    // std::cout << "fAUPGain = " << fAUPGain << "\n";
	return fAUPGain;
}

void ConditionalGain::updateMemoization(std::unordered_set<ll> const &X, ll item) {
    if (X.find(item)!=X.end()) {
		return;
	}
    if (privateSet.find(item) != privateSet.end()) {
        return;
    } else {
        //fAUP->updateMemoization(unionPreComputeSet, item);
        f.updateMemoization(unionPreComputeSet, item);
        unionPreComputeSet.insert(item);
    }
}

 std::vector<std::pair<ll, double>> ConditionalGain::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true) {
 	// std::cout << "ConditionalGain maximize\n";
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

 std::unordered_set<ll> ConditionalGain::getEffectiveGroundSet() {
 	std::unordered_set<ll> effectiveGroundSet;
 	effectiveGroundSet.reserve(n);
 	for (ll i = 0; i < n; ++i){
 		effectiveGroundSet.insert(i); 
 	}
 	return effectiveGroundSet;
 }

void ConditionalGain::clearMemoization()
{
    unionPreComputeSet.clear();
    unionPreComputeSet = privateSet;
    //fAUP->setMemoization(privateSet); 
    f.setMemoization(privateSet); 
}

void ConditionalGain::setMemoization(std::unordered_set<ll> const &X) 
{
    unionPreComputeSet.clear();
    unionPreComputeSet = set_union(privateSet, X);
    //fAUP->setMemoization(unionPreComputeSet);
    f.setMemoization(unionPreComputeSet);
}
