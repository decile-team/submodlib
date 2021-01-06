#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"../optimizers/NaiveGreedyOptimizer.h"
#endif

#ifndef SETFUNCTION_H
#define SETFUNCTION_H
#include"SetFunction.h"
#endif


#ifndef FACILITYLOCATION_H
#define FACILITYLOCATION_H
#include "FacilityLocation.h"
#include "DisparitySum.h"
#endif

typedef long long int ll;

class ClusteredFunction : public SetFunction
{
    ll n;
    std::set<ll> effectiveGroundSet;

    ll num_cluster;
    std::string fun_name;
    std::vector<std::set<ll>>clusters; //contains original data indicies
	std::vector<std::set<ll>>clusters_translated; //contains cluster indicies consistent with indicies in kernel corrosponding to cluster 
    std::vector<std::vector<std::vector<float>>>v_k_cluster;//vector which contains dense similarity matrix for each cluster
	std::vector<ll>v_k_ind;
    
    std::vector<SetFunction* >v_fun;
    std::vector<ll>clusterIDs;

    public:
    ClusteredFunction(ll n_, std::string fun_name_, std::vector<std::set<ll>>clusters_, std::vector<std::vector<std::vector<float>>>v_k_cluster_, std::vector<ll>v_k_ind_ );
    float evaluate(std::set<ll> X);
	float evaluateSequential(std::set<ll> X);
	float marginalGain(std::set<ll> X, ll item);
	float marginalGainSequential(std::set<ll> X, ll item);
	void sequentialUpdate(std::set<ll> X, ll item);
    std::set<ll> getEffectiveGroundSet();
    std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);

    friend std::set<ll> translate_X(std::set<ll>X, ClusteredFunction obj, ll cluster_id);
};

std::set<ll> translate_X(std::set<ll>X, ClusteredFunction obj, ll cluster_id);