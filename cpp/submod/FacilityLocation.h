#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"../optimizers/NaiveGreedyOptimizer.h"
#endif

#ifndef SETFUNCTION_H
#define SETFUNCTION_H
#include"../SetFunction.h"
#endif

#ifndef SPARSEUTILS_H
#define SPARSEUTILS_H
#include"../utils/sparse_utils.h"
#endif

typedef long long int ll;

class FacilityLocation :public SetFunction
{
protected:

	ll n; //size of ground set
	ll n_master; //size of master set
	std::string mode; //can be dense, sparse or clustered
	bool partial; //if masked implementation is desired, relevant to be used in ClusteredFunction
	bool separateMaster; //if master set is separate from ground set
	std::set<ll> effectiveGroundSet; //effective ground set considering mask if partial = true
	std::set<ll> masterSet; //set of items in master set
	ll numEffectiveGroundset; //size of effective ground set
	std::map<ll, ll> originalToPartialIndexMap;
	
	std::vector<std::vector<float>>denseKernel; //size n_master X n
	SparseSim sparseKernel = SparseSim(); 

	//Specific to cluster mode only
	ll num_clusters;
	std::vector<std::set<ll>>clusters; //vector of clusters (where each cluster is taken as a set of datapoint indices in that cluster), size = num_clusters
	std::vector<ll>clusterIDs;//maps index of a datapoint to the ID of cluster which it belongs to, size = n
	std::vector<std::vector<std::vector<float>>>clusterKernels;//vector which contains dense similarity matrices for each cluster, size = num_clusters
	std::vector<ll>clusterIndexMap; //mapping from datapont index to index in cluster kernel, size = n

    //memoized statistics
	std::vector<float> similarityWithNearestInEffectiveX; //for each i in master set it contains max(i, effectiveX), size = n_master
	std::vector<std::set<ll>>relevantX; //for each cluster C_i it contains X \cap C_i, size = num_clusters
	std::vector<float>clusteredSimilarityWithNearestInRelevantX;//for every element in ground set, this vector contains its maximum similarity with items in X \cap C_i where C_i is the cluster which this element belongs to, size = n

public:

	FacilityLocation();

	//For dense mode
	FacilityLocation(ll n_, std::vector<std::vector<float>>denseKernel_, bool partial_, std::set<ll> ground_, bool separateMaster_);
	
	//For sparse mode
	FacilityLocation(ll n_, std::vector<float>arr_val, std::vector<ll>arr_count, std::vector<ll>arr_col);

	//For clustered mode
	FacilityLocation(ll n_, std::vector<std::set<ll>>clusters_, std::vector<std::vector<std::vector<float>>>clusterKernels_, std::vector<ll>clusterIndexMap_);


	float evaluate(std::set<ll> X);
	float evaluateWithMemoization(std::set<ll> X);
	float marginalGain(std::set<ll> X, ll item);
	float marginalGainWithMemoization(std::set<ll> X, ll item);
	void updateMemoization(std::set<ll> X, ll item);
	std::set<ll> getEffectiveGroundSet();
	std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);
	void cluster_init(ll n_, std::vector<std::vector<float>>denseKernel_, std::set<ll> ground_, bool partial);
	void clearMemoization();
	void setMemoization(std::set<ll> X);

	friend float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
	friend float get_max_sim_sparse(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
	friend float get_max_sim_cluster(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj, ll cluster_id);
};


float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
float get_max_sim_sparse(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
float get_max_sim_cluster(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj, ll cluster_id);

