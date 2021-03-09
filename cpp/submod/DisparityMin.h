#include "../optimizers/NaiveGreedyOptimizer.h"
#include "SetFunction.h"
#include"../utils/sparse_utils.h"

class DisparityMin : public SetFunction {
    // Generic stuff
    ll n;  // number of datapoints in ground set
    std::string mode;
    ll num_neighbors;
    bool partial;
    std::unordered_set<ll> effectiveGroundSet;
    ll numEffectiveGroundset;

    // Main kernels and containers for all 3 modes
    std::vector<std::vector<float>> k_dense;
    SparseSim k_sparse = SparseSim();

    float currentSum;

   public:
    DisparityMin();

    // For dense similarity matrix
    DisparityMin(ll n_, std::string mode_,
                 std::vector<std::vector<float>> k_dense_, ll num_neighbors_,
                 bool partial_, std::unordered_set<ll> ground_);

    // For sparse similarity matrix
    DisparityMin(ll n_, std::string mode_, std::vector<float> arr_val,
                 std::vector<ll> arr_count, std::vector<ll> arr_col,
                 ll num_neighbors_, bool partial_, std::unordered_set<ll> ground_);

    float evaluate(std::unordered_set<ll> X);
    float evaluateWithMemoization(std::unordered_set<ll> X);
    float marginalGain(std::unordered_set<ll> X, ll item);
    float marginalGainWithMemoization(std::unordered_set<ll> X, ll item);
    void updateMemoization(std::unordered_set<ll> X, ll item);
    std::unordered_set<ll> getEffectiveGroundSet();
    std::vector<std::pair<ll, float>> maximize(std::string, float budget,
                                               bool stopIfZeroGain,
                                               bool stopIfNegativeGain,
                                               bool verbosity);
    void cluster_init(ll n_, std::vector<std::vector<float>> k_dense_,
                      std::unordered_set<ll> ground_);
    void clearMemoization();
    void setMemoization(std::unordered_set<ll> X);

    friend float get_sum_dense(std::unordered_set<ll> dataset_ind, DisparityMin obj);
    friend float get_sum_sparse(std::unordered_set<ll> dataset_ind, DisparityMin obj);
};

float get_sum_dense(std::unordered_set<ll> dataset_ind, DisparityMin obj);
float get_sum_sparse(std::unordered_set<ll> dataset_ind, DisparityMin obj);
