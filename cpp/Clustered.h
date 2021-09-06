#include"SetFunction.h"
#include "submod/FacilityLocation.h"
#include "submod/DisparitySum.h"
#include "submod/LogDeterminant.h"
#include "submod/DisparityMin.h"
#include "submod/GraphCut.h"
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
    float lambda;
    
    std::vector<SetFunction* >mixture;
    std::vector<ll>clusterIDs;

    public:
    Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>> const &clusters_, std::vector<std::vector<std::vector<float>>> const &clusterKernels_, std::vector<ll> const &clusterIndexMap_, float lambda_);
    Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>> const &clusters_, std::vector<std::vector<float>> const &denseKernel_, float lambda_);
    double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
    std::unordered_set<ll> getEffectiveGroundSet();
    //std::vector<std::pair<ll, double>> maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose, bool showProgress);
    void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
    // Clustered* clone();

    friend std::unordered_set<ll> translate_X(std::unordered_set<ll> const &X, Clustered const &obj, ll cluster_id);
};

std::unordered_set<ll> translate_X(std::unordered_set<ll> const &X, Clustered const &obj, ll cluster_id);
