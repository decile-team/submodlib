#ifndef FEATUREBASED_H
#define FEATUREBASED_H

#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include <unordered_set>

class FeatureBased :public SetFunction
{
protected:
    int numFeatures;
	ll n; //size of ground set
    std::vector<std::vector<std::pair<int, float>>> groundSetFeatures;
	std::vector<float> featureWeights;
	double transform(double val);
	std::vector<double> sumOfFeaturesAcrossX; //memoized statistics for X
public:
    enum Type {
        squareRoot,
		inverse,
		logarithmic
	};
	Type type;
	FeatureBased(ll n_, Type type_, std::vector<std::vector<std::pair<int, float>>> const &groundSetFeatures_, int numFeatures_, std::vector<float> const& featureWeights_);

	double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	// std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
	void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
	// FeatureBased* clone();
};
#endif
