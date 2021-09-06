
#include<iostream>
#include<set>
#include<vector>
#include<utility>
#include<string>
#include"SetFunction.h"
#include"optimizers/NaiveGreedyOptimizer.h"
#include"optimizers/LazyGreedyOptimizer.h"
#include"optimizers/StochasticGreedyOptimizer.h"
#include"optimizers/LazierThanLazyGreedyOptimizer.h"

double SetFunction::evaluate(std::unordered_set<ll> const &X){}
double SetFunction::evaluateWithMemoization(std::unordered_set<ll> const &X){}
double SetFunction::marginalGain(std::unordered_set<ll> const &X, ll item){}
double SetFunction::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks){}
void SetFunction::updateMemoization(std::unordered_set<ll> const &X, ll item){}
std::unordered_set<ll> SetFunction::getEffectiveGroundSet(){}
std::vector<std::pair<ll, double>> SetFunction::maximize(std::string optimizer, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress, const std::vector<float>& costs, bool costSensitiveGreedy){
	if(optimizer == "NaiveGreedy") {
		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress, costs, costSensitiveGreedy);
	} else if(optimizer == "LazyGreedy") {
        return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress, costs, costSensitiveGreedy);
	} else if(optimizer == "StochasticGreedy") {
        return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress, costs, costSensitiveGreedy);
	} else if(optimizer == "LazierThanLazyGreedy") {
        return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress, costs, costSensitiveGreedy);
	} else {
		std::cout << "Invalid Optimizer" << std::endl;
	}
}
void SetFunction::cluster_init(ll n_, std::vector<std::vector<float>> const &k_dense_, std::unordered_set<ll> const &ground_, bool partial, float lambda){}
void SetFunction::setMemoization(std::unordered_set<ll> const &X){}
void SetFunction::clearMemoization(){}
SetFunction * SetFunction::clone() {return NULL;}