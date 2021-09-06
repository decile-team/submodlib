#ifndef PSCMI_H
#define PSCMI_H

#include "../submod/ProbabilisticSetCover.h"
#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include <unordered_set>

class ProbabilisticSetCoverMutualInformation :public SetFunction
{
protected:
    int numConcepts;
	ll n; //size of ground set
	std::unordered_set<int> Q; //set of concept indices in query set
    std::vector<std::vector<float>> groundSetConceptProbabilities;
	std::vector<float> conceptWeights;
	std::vector<float> conceptWeightsQ;
	ProbabilisticSetCover* pscQ;
public:
	ProbabilisticSetCoverMutualInformation(ll n_, int numConcepts_,std::vector<std::vector<float>> const &groundSetConceptProbabilities_,  std::vector<float> const& conceptWeights_, std::unordered_set<int> const & Q_);
	double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	//std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
	void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
	// ProbabilisticSetCoverMutualInformation* clone();
};
#endif
