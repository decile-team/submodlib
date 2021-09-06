#ifndef FACILITYLOCATION_H
#define FACILITYLOCATION_H

#include"../optimizers/NaiveGreedyOptimizer.h"
#include"../optimizers/LazyGreedyOptimizer.h"
#include"../optimizers/StochasticGreedyOptimizer.h"
#include"../optimizers/LazierThanLazyGreedyOptimizer.h"
#include"../SetFunction.h"
#include"../utils/sparse_utils.h"
#include <unordered_set>

class FacilityLocation :public SetFunction
{
protected:
	ll n; //size of ground set
	ll n_master; //size of master set
	//std::string mode; //can be dense, sparse or clustered
	enum Mode {
        dense,
		sparse,
		clustered	
	};
	Mode mode;
	bool partial; //if masked implementation is desired, relevant to be used in ClusteredFunction
	bool separateMaster; //if master set is separate from ground set
	std::unordered_set<ll> effectiveGroundSet; //effective ground set considering mask if partial = true
	std::unordered_set<ll> masterSet; //set of items in master set
	ll numEffectiveGroundset; //size of effective ground set
	std::map<ll, ll> originalToPartialIndexMap;
	
	std::vector<std::vector<float>>denseKernel; //size n_master X n
	SparseSim sparseKernel = SparseSim(); 

	//Specific to cluster mode only
	ll num_clusters;
	std::vector<std::unordered_set<ll>>clusters; //vector of clusters (where each cluster is taken as a set of datapoint indices in that cluster), size = num_clusters
	std::vector<ll>clusterIDs;//maps index of a datapoint to the ID of cluster which it belongs to, size = n
	std::vector<std::vector<std::vector<float>>>clusterKernels;//vector which contains dense similarity matrices for each cluster, size = num_clusters
	std::vector<ll>clusterIndexMap; //mapping from datapont index to index in cluster kernel, size = n

    //memoized statistics
	std::vector<float> similarityWithNearestInEffectiveX; //for each i in master set it contains max(i, effectiveX), size = n_master
	std::vector<std::unordered_set<ll>>relevantX; //for each cluster C_i it contains X \cap C_i, size = num_clusters
	std::vector<float>clusteredSimilarityWithNearestInRelevantX;//for every element in ground set, this vector contains its maximum similarity with items in X \cap C_i where C_i is the cluster which this element belongs to, size = n

public:

	FacilityLocation();

	//For dense mode with kernel
	FacilityLocation(ll n_, std::vector<std::vector<float>> const &denseKernel_, bool partial_, std::unordered_set<ll> const &ground_, bool separateMaster_);

    //For dense mode without kernel
	FacilityLocation(ll n_, std::vector<std::vector<float>> &data, std::vector<std::vector<float>> &data_master, bool separateMaster_, std::string metric);

	//For sparse mode
	FacilityLocation(ll n_, std::vector<float> const &arr_val, std::vector<ll> const &arr_count, std::vector<ll> const &arr_col);

	//For clustered mode
	FacilityLocation(ll n_, std::vector<std::unordered_set<ll>> const &clusters_, std::vector<std::vector<std::vector<float>>> const &clusterKernels_, std::vector<ll> const &clusterIndexMap_);

	FacilityLocation(const FacilityLocation& f);
    FacilityLocation* clone();

	double evaluate(std::unordered_set<ll> const &X);
	double evaluateWithMemoization(std::unordered_set<ll> const &X);
	double marginalGain(std::unordered_set<ll> const &X, ll item);
	double marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks=true);
	void updateMemoization(std::unordered_set<ll> const &X, ll item);
	std::unordered_set<ll> getEffectiveGroundSet();
	void cluster_init(ll n_, std::vector<std::vector<float>> const &denseKernel_, std::unordered_set<ll> const &ground_, bool partial, float lambda);
	void clearMemoization();
	void setMemoization(std::unordered_set<ll> const &X);
	// FacilityLocation* clone();

	friend float get_max_sim_dense(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj);
	friend float get_max_sim_sparse(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj);
	friend float get_max_sim_cluster(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj, ll cluster_id);
};


float get_max_sim_dense(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj);
float get_max_sim_sparse(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj);
float get_max_sim_cluster(ll datapoint_ind, std::unordered_set<ll> const &dataset_ind, FacilityLocation &obj, ll cluster_id);

#endif
