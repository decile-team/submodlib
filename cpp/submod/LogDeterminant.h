//Implementation of Algorithm 1 in https://proceedings.neurips.cc/paper/2018/file/dbbf603ff0e99629dda5d75b6f75f966-Paper.pdf

#ifndef LOGDETERMINANT_H
#define LOGDETERMINANT_H
#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include"../utils/sparse_utils.h"
#include <unordered_set>


class LogDeterminant : public SetFunction {
    protected:
    ll n;  
    enum Mode {
        dense,
		sparse,
		clustered	
	};
	Mode mode;
    bool partial;
    std::unordered_set<ll> effectiveGroundSet;
    ll numEffectiveGroundset;
    std::map<ll, ll> originalToPartialIndexMap;

    std::vector<std::vector<float>> denseKernel;
    SparseSim sparseKernel = SparseSim();

    std::vector<std::vector<double>> memoizedC; //memoized ci
    std::vector<double> memoizedD; //memoized di

    int prevItem;
    double prevDetVal;

    double lambda;

   public:
    LogDeterminant();
    // For dense similarity matrix
    LogDeterminant(ll n_, std::vector<std::vector<float>> const &denseKernel_, bool partial_, std::unordered_set<ll> const &ground_, double lambda_);
    // For sparse similarity matrix
    LogDeterminant(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col, double lambda_);
    LogDeterminant(const LogDeterminant& f);
    LogDeterminant* clone();
    double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	// std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
	void cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda);
    void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
};
#endif