#ifndef GRAPHCUT_H
#define GRAPHCUT_H

#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include"../utils/sparse_utils.h"
#include <unordered_set>

class GraphCut :public SetFunction {
protected:
	ll n; //size of ground set
	ll n_master; //size of master set
	enum Mode {
        dense,
		sparse,
		clustered	
	};
	Mode mode;
	bool partial; //if masked implementation is desired, relevant to be used in ClusteredFunction
	bool separateMaster; //if master set is separate from ground set
	std::unordered_set<ll> effectiveGroundSet; //effective ground set considering mask if partial = true
	std::unordered_set<ll> masterSet; //set of items in master set
	ll numEffectiveGroundset; //size of effective ground set
	std::map<ll, ll> originalToPartialIndexMap;
	std::vector<std::vector<float>>masterGroundKernel; //size n_master X n
	std::vector<std::vector<float>>groundGroundKernel; //size n X n
	SparseSim sparseKernel = SparseSim(); 

	std::vector<double> totalSimilarityWithMaster;
	float lambda;
	//memoized statistics
	std::vector<double> totalSimilarityWithSubset;
	
public:
	GraphCut();
	//For dense mode with master = ground
	GraphCut(ll n_, std::vector<std::vector<float>> const &masterGroundKernel_, bool partial_, std::unordered_set<ll> const &ground_, float lambda);
	//For dense mode with master != ground
	GraphCut(ll n_, std::vector<std::vector<float>> const &masterGroundKernel_, std::vector<std::vector<float>> const &groundGroundKernel_, float lambda_);
	//For sparse mode
	GraphCut(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col, float lambda_);

	double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	// std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
	void cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda_);
	void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
	// GraphCut* clone();
};
#endif
