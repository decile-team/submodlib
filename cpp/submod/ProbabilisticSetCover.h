#ifndef PSETCOVER_H
#define PSETCOVER_H

#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include <unordered_set>

class ProbabilisticSetCover :public SetFunction
{
protected:
    int numConcepts;
	ll n; //size of ground set
    std::vector<std::vector<float>> groundSetConceptProbabilities;
	std::vector<float> conceptWeights;
	std::vector<double> probOfConceptsCoveredByX; //memoized statistics for X
public:
	ProbabilisticSetCover(ll n_, std::vector<std::vector<float>> const &groundSetConceptProbabilities, int numConcepts_, std::vector<float> const& conceptWeights_);

	double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	// std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
	void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
	// ProbabilisticSetCover* clone();
};
#endif
