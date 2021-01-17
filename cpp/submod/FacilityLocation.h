

/*
Implementation decisions.
1) Considering the possibility of very large datasets, its safer to use long long int (alias ll) in place of int (for storing size/index of data)

2) Containers like X, groundset, effectiveGroundSet etc (which contain index of datapoints) have been implemented as set (instead of vector).
This is because in C++, set container is implemented as red-black tree and thus search operations happen in log(n) time which is beneficial
for functions like marginalGain(), sequentialUpdate() etc that require such search operations frequently.
If we use vectors then for efficiency we would have an additional responsibility of ensuring that they are sorted. Thus,
set is a more natural choice here

3) For sparse mode, constructor will accept sparse matrix as a collection of 3 component vectors (for csr) and use them to instantiate
a sparse matrix object either using a custom utility class or using some high performance library like boost.

*/
#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"../optimizers/NaiveGreedyOptimizer.h"
#endif

#ifndef SETFUNCTION_H
#define SETFUNCTION_H
#include"SetFunction.h"
#endif

#ifndef SPARSEUTILS_H
#define SPARSEUTILS_H
#include"../utils/sparse_utils.h"
#endif

typedef long long int ll;

class FacilityLocation :public SetFunction
{

	//Generic stuff
	ll n; //number of datapoints in ground set
	ll n_master; //number of datapoints in master set
	std::string mode;
	ll num_neighbors;
	bool partial;
	bool seperateMaster;
	std::set<ll> effectiveGroundSet;
	std::set<ll> masterSet;
	ll numEffectiveGroundset;
	
	//Main kernels and containers for all 3 modes
	std::vector<std::vector<float>>k_dense;
	SparseSim k_sparse = SparseSim(); 
	std::vector<std::set<ll>>clusters; //vector of clusters (where each cluster is taken as a set of datapoint index)
	std::vector<std::vector<std::vector<float>>>v_k_cluster;//vector which contains dense similarity matrix for each cluster
	std::vector<ll>v_k_ind;

	//Specific to dense and sparse mode only
	std::vector<float> similarityWithNearestInEffectiveX;//memoization vector for dense and sparse mode
	
	//Specific to cluster mode only
	ll num_cluster;
	std::vector<float>clusteredSimilarityWithNearestInRelevantX;//memoization vector for cluster mode
	std::vector<ll>clusterIDs;//maps index of a datapoint to the ID of cluster where its present.
	std::vector<std::set<ll>>relevantX;
	


public:

	FacilityLocation();

	//For dense similarity matrix
	FacilityLocation(ll n_, std::string mode_, std::vector<std::vector<float>>k_dense_, ll num_neighbors_, bool partial_, std::set<ll> ground_, bool seperateMaster_);
	
	//For sparse similarity matrix
	FacilityLocation(ll n_, std::string mode_, std::vector<float>arr_val, std::vector<ll>arr_count, std::vector<ll>arr_col, ll num_neighbors_, bool partial_, std::set<ll> ground_);

	//For cluster mode
	FacilityLocation(ll n_, std::string mode_, std::vector<std::set<ll>>clusters_, std::vector<std::vector<std::vector<float>>>v_k_cluster_, std::vector<ll>v_k_ind_, ll num_neighbors_, bool partial_, std::set<ll> ground_);


	float evaluate(std::set<ll> X);
	float evaluateSequential(std::set<ll> X);
	float marginalGain(std::set<ll> X, ll item);
	float marginalGainSequential(std::set<ll> X, ll item);
	void sequentialUpdate(std::set<ll> X, ll item);
	std::set<ll> getEffectiveGroundSet();
	std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);
	void cluster_init(ll n_, std::vector<std::vector<float>>k_dense_, std::set<ll> ground_);
	void clearPreCompute();


	friend float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
	friend float get_max_sim_sparse(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
	friend float get_max_sim_cluster(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj, ll cluster_id);
};


float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
float get_max_sim_sparse(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
float get_max_sim_cluster(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj, ll cluster_id);

