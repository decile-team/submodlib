#ifndef FLMI_H
#define FLMI_H
#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include <unordered_set>

class FacilityLocationMutualInformation : public SetFunction {
    protected:
    ll n;  
    int numQueries;
    float magnificationLambda;
    // std::unordered_set<int> querySet;
    // std::map<int, int> indexMap;
    std::vector<std::vector <float> > kernelImage;   //n X n
    std::vector<std::vector <float> > kernelQuery;   //n X numQueries

    std::vector<float> similarityWithNearestInX;
    std::vector<float> qMaxMod;

   public:
    FacilityLocationMutualInformation(ll n_, int numQueries_, std::vector<std::vector<float>> const &kernelImage_, std::vector<std::vector<float>> const &kernelQuery_, float magnificationLambda_);

    double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
    std::unordered_set<ll> getEffectiveGroundSet();
	//std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
    void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
    // FacilityLocationMutualInformation* clone();
};
#endif