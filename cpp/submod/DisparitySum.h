#ifndef DISPARITYSUM_H
#define DISPARITYSUM_H
#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../SetFunction.h"
#include"../utils/sparse_utils.h"
#include <unordered_set>


class DisparitySum : public SetFunction {
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

    double currentSum;  //by convention, 0 for size 0 and size 1 sets

   public:
    DisparitySum();
    // For dense similarity matrix
    DisparitySum(ll n_, std::vector<std::vector<float>> const &denseKernel_, bool partial_, std::unordered_set<ll> const &ground_);
    // For sparse similarity matrix
    DisparitySum(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col);

    double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	//std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
	void cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda);
    void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
    // DisparitySum* clone();

    friend double get_sum_dense(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj);
    friend double get_sum_sparse(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj);
};

double get_sum_dense(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj);
double get_sum_sparse(std::unordered_set<ll> const &dataset_ind, DisparitySum &obj);
#endif