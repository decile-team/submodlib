#ifndef SETCOVERCMI_H
#define SETCOVERCMI_H

#include "../submod/SetCover.h"
#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include <unordered_set>

class SetCoverConditionalMutualInformation :public SetFunction
{
protected:
    int numConcepts;
	ll n; //size of ground set
	std::unordered_set<int> Q; //set of concept indices in query set
	std::unordered_set<int> P; //set of concept indices in private set
    std::vector<std::unordered_set<int>> coverSet;
	std::vector<float> conceptWeights;
	std::vector<std::unordered_set<int>> coverSetQMinusP;
	SetCover* scQMinusP;

public:
	SetCoverConditionalMutualInformation(ll n_, std::vector<std::unordered_set<int>> const &coverSet_, int numConcepts_, std::vector<float> const& conceptWeights_, std::unordered_set<int> const & Q_, std::unordered_set<int> const & P_);
	double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	//std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
	void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
	// SetCoverConditionalMutualInformation* clone();
};
#endif
