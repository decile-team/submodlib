
#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include "../optimizers/NaiveGreedyOptimizer.h"
#endif

#ifndef SETFUNCTION_H
#define SETFUNCTION_H
#include "SetFunction.h"
#endif

#ifndef SPARSEUTILS_H
#define SPARSEUTILS_H
#include"../utils/sparse_utils.h"
#endif

typedef long long int ll;

class DisparitySum : public SetFunction {
    // Generic stuff
    ll n;  // number of datapoints in ground set
    std::string mode;
    ll num_neighbors;
    bool partial;
    std::set<ll> effectiveGroundSet;
    ll numEffectiveGroundset;

    // Main kernels and containers for all 3 modes
    std::vector<std::vector<float>> k_dense;
    SparseSim k_sparse = SparseSim();

    float currentSum;

   public:
    DisparitySum();

    // For dense similarity matrix
    DisparitySum(ll n_, std::string mode_,
                 std::vector<std::vector<float>> k_dense_, ll num_neighbors_,
                 bool partial_, std::set<ll> ground_);

    // For sparse similarity matrix
    DisparitySum(ll n_, std::string mode_, std::vector<float> arr_val,
                 std::vector<ll> arr_count, std::vector<ll> arr_col,
                 ll num_neighbors_, bool partial_, std::set<ll> ground_);

    float evaluate(std::set<ll> X);
    float evaluateSequential(std::set<ll> X);
    float marginalGain(std::set<ll> X, ll item);
    float marginalGainSequential(std::set<ll> X, ll item);
    void sequentialUpdate(std::set<ll> X, ll item);
    std::set<ll> getEffectiveGroundSet();
    std::vector<std::pair<ll, float>> maximize(std::string, float budget,
                                               bool stopIfZeroGain,
                                               bool stopIfNegativeGain,
                                               bool verbosity);
    void cluster_init(ll n_, std::vector<std::vector<float>> k_dense_,
                      std::set<ll> ground_);
    void clearPreCompute();
    void setMemoization(std::set<ll> X);

    friend float get_sum_dense(std::set<ll> dataset_ind, DisparitySum obj);
    friend float get_sum_sparse(std::set<ll> dataset_ind, DisparitySum obj);
};

float get_sum_dense(std::set<ll> dataset_ind, DisparitySum obj);
float get_sum_sparse(std::set<ll> dataset_ind, DisparitySum obj);
