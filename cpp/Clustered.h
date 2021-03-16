#include"optimizers/NaiveGreedyOptimizer.h"
#include"optimizers/LazyGreedyOptimizer.h"
#include"SetFunction.h"
#include "submod/FacilityLocation.h"
#include "submod/DisparitySum.h"
#include <unordered_set>

class Clustered : public SetFunction
{
    ll n;
    std::unordered_set<ll> effectiveGroundSet;

    ll num_clusters;
    std::string function_name;
    std::vector<std::unordered_set<ll>>clusters; //contains original data indicies
	std::vector<std::unordered_set<ll>>clusters_translated; //contains cluster indicies consistent with indicies in kernel corrosponding to cluster //TODO: this is redundant because the translated indices are going to be just 0, 1, 2,... for each cluster
    std::vector<std::vector<std::vector<float>>>clusterKernels;//vector which contains dense similarity matrix for each cluster
	std::vector<ll>clusterIndexMap;
    std::vector<std::vector<float>> denseKernel;
    enum Mode {
        single,
		multi	
	};
	Mode mode;
    
    std::vector<SetFunction* >mixture;
    std::vector<ll>clusterIDs;

    public:
    Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>> const &clusters_, std::vector<std::vector<std::vector<float>>> const &clusterKernels_, std::vector<ll> const &clusterIndexMap_ );
    Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>> const &clusters_, std::vector<std::vector<float>> const &denseKernel_);
    float evaluate(std::unordered_set<ll> const &X);
	float evaluateWithMemoization(std::unordered_set<ll> const &X);
	float marginalGain(std::unordered_set<ll> const &X, ll item);
	float marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
    std::unordered_set<ll> getEffectiveGroundSet();
    std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbose);
    void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);

    friend std::unordered_set<ll> translate_X(std::unordered_set<ll> const &X, Clustered const &obj, ll cluster_id);
};

std::unordered_set<ll> translate_X(std::unordered_set<ll> const &X, Clustered const &obj, ll cluster_id);
