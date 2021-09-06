#ifndef GCCONDGAIN_H
#define GCCONDGAIN_H
#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include <unordered_set>

class GraphCutConditionalGain : public SetFunction {
    protected:
    ll n;  
    int numPrivates;
    float privacyHardness;
    std::vector<std::vector <float> > kernelImage;   //n X n
    std::vector<std::vector <float> > kernelPrivate;   //n X numQueries
    float lambda;
    std::vector<double> totalSimilarityWithMaster;
    std::vector<float> pSum;
    std::vector<double> totalSimilarityWithSubset; //memoized statistic for GC

   public:
    GraphCutConditionalGain(ll n_, int numPrivates_, std::vector<std::vector<float>> const &kernelImage_, std::vector<std::vector<float>> const &kernelPrivate_, float privacyHardness_, float lambda_);

    double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
    std::unordered_set<ll> getEffectiveGroundSet();
	//std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
    void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
    // GraphCutConditionalGain* clone();
};
#endif