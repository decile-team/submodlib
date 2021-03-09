#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"optimizers/NaiveGreedyOptimizer.h"
#endif

#ifndef SETFUNCTION_H
#define SETFUNCTION_H
#include"SetFunction.h"
#endif


#ifndef FACILITYLOCATION_H
#define FACILITYLOCATION_H
#include "submod/FacilityLocation.h"
#include "submod/DisparitySum.h"
#endif

typedef long long int ll;

class Clustered : public SetFunction
{
    ll n;
    std::set<ll> effectiveGroundSet;

    ll num_clusters;
    std::string function_name;
    std::vector<std::set<ll>>clusters; //contains original data indicies
	std::vector<std::set<ll>>clusters_translated; //contains cluster indicies consistent with indicies in kernel corrosponding to cluster //TODO: this is redundant because the translated indices are going to be just 0, 1, 2,... for each cluster
    std::vector<std::vector<std::vector<float>>>clusterKernels;//vector which contains dense similarity matrix for each cluster
	std::vector<ll>clusterIndexMap;
    std::vector<std::vector<float>> denseKernel;
    std::string mode;
    
    std::vector<SetFunction* >mixture;
    std::vector<ll>clusterIDs;

    public:
    Clustered(ll n_, std::string function_name_, std::vector<std::set<ll>>clusters_, std::vector<std::vector<std::vector<float>>>clusterKernels_, std::vector<ll>clusterIndexMap_ );
    Clustered(ll n_, std::string function_name_, std::vector<std::set<ll>>clusters_, std::vector<std::vector<float>>denseKernel_);
    float evaluate(std::set<ll> X);
	float evaluateWithMemoization(std::set<ll> X);
	float marginalGain(std::set<ll> X, ll item);
	float marginalGainWithMemoization(std::set<ll> X, ll item);
	void updateMemoization(std::set<ll> X, ll item);
    std::set<ll> getEffectiveGroundSet();
    std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);
    void clearMemoization();
	void setMemoization(std::set<ll> X);

    friend std::set<ll> translate_X(std::set<ll>X, Clustered obj, ll cluster_id);
};

std::set<ll> translate_X(std::set<ll>X, Clustered obj, ll cluster_id);