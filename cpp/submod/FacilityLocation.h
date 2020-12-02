/*
Implementation decisions.
1) Considering the possibility of very large datasets, its safer to use long long int (alias ll) in place of int (for storing size/index of data)

2) Containers like X, groundset, effectiveGroundSet etc (which contain index of datapoints) have been implemented as set (instead of vector).
This is because in C++, set container is implemented as red-black tree and thus search operations happen in log(n) time which is beneficial
for functions like marginalGain(), sequentialUpdate() etc that require such search operations frequently.
If we use vectors then for efficiency we would have an additional responsibility of ensuring that they are sorted. Thus,
set is a more natural choice here

3) For sparse mode, constructor will accept sparse matrix as a map of 3 component vectors (for csr) and use them to instantiate
a sparse matrix object either using a custom utility class or using some high performance library like boost.

*/

typedef long long int ll;

class FacilityLocation
{

	ll n; 
	std::string mode;
	std::vector<std::vector<float>>k_dense;
	//class_sparse k_sparse; 
	std::vector<std::set<ll>>clusters; //vector of clusters (where each cluster is taken as a set of datapoint index)
	ll num_neighbors;
	bool partial;
	std::set<ll> effectiveGroundSet;
	ll numEffectiveGroundset;
	std::vector<float> similarityWithNearestInEffectiveX;
	//std::map<std::vector<float>, ll>map_data_to_ind;

public:
	//constructor(no_of_elem_in_ground, mode, sim_matrix or cluster, num_neigh, partial, ground_subset )

	//For dense similarity matrix
	FacilityLocation(ll n_, std::string mode_, std::vector<std::vector<float>>k_dense_, ll num_neighbors_, bool partial_, std::set<ll> ground_);
	
	//For sparse similarity matrix
	//FacilityLocation(ll n_, std::string mode_, std::map<std::string, std::vector<float>>k_sparse_, ll num_neighbors_, bool partial_, std::set<ll> ground_);

	//For cluster mode
	FacilityLocation(ll n_, std::string mode_, std::vector<std::set<ll>>clusters_, ll num_neighbors_, bool partial_, std::set<ll> ground_);


	float evaluate(std::set<ll> X);
	float evaluateSequential(std::set<ll> X);
	float marginalGain(std::set<ll> X, ll item);
	float marginalGainSequential(std::set<ll> X, ll item);
	void sequentialUpdate(std::set<ll> X, ll item);
	std::set<ll> getEffectiveGroundSet();

	friend float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
};


float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj);
