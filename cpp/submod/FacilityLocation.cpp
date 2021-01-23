/*
Implementation decisions.
1) Considering the possibility of very large datasets, its safer to use long long int (alias ll) in place of int (for storing size/index of data)

2) Containers like X, groundset, effectiveGroundSet etc (which contain index of datapoints) have been implemented as set (instead of vector).
This is because in C++, set container is implemented as red-black tree and thus search operations happen in log(n) time which is beneficial
for functions like marginalGain(), updateMemoization() etc that require such search operations frequently.
If we use vectors then for efficiency we would have an additional responsibility of ensuring that they are sorted. Thus,
set is a more natural choice here

3) For sparse mode, constructor will accept sparse matrix as a collection of 3 component vectors (for csr) and use them to instantiate
a sparse matrix object either using a custom utility class or using some high performance library like boost.

Possible Improvement.
If order of elements in X, groundset, effectiveGroundSet etc doesn't matter, we can improve the performance further by using unordered_set instead of 
set. unordered_set is implemented using hashmap and has an average search complexity of O(1). Although, if this is done then we also need to write
a custom function for "Intersection" because set_intersection() only work on containers with sorted data. Also, to avoid excessive rehashing, we will
have to reserve a certain number of buckets in advance.


*/

#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"FacilityLocation.h"

typedef long long int ll;

//Note to self: Migrate all parameter related sanity/error checks from C++ FL to Python FL

FacilityLocation::FacilityLocation(){}

//For dense mode
FacilityLocation::FacilityLocation(ll n_, std::string mode_, std::vector<std::vector<float>>k_dense_, ll num_neighbors_, bool partial_, std::set<ll> ground_, bool seperateMaster_)
{
	if (mode_ != "dense") 
	{
		std::cerr << "Error: Incorrect mode specified for the provided dense similarity matrix\n";
		return;
	}

	if (k_dense_.size() == 0)
	{
		std::cerr << "Error: Empty similarity matrix\n";
		return;
	}

	n = n_;
	mode = mode_;
	k_dense = k_dense_;
	num_neighbors = num_neighbors_;
	partial = partial_;
	seperateMaster = seperateMaster_;
	//Populating effectiveGroundSet
	if (partial == true)
	{
		effectiveGroundSet = ground_;
	}
	else
	{
		for (ll i = 0; i < n; ++i)
		{
			effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
		}
	}
	
	//Populating masterSet
	if(mode=="dense" && seperateMaster==true)
	{
		n_master = k_dense.size();	
		for (ll i = 0; i < n_master; ++i)
		{
			masterSet.insert(i); //each insert takes O(log(n)) time
		}
	}
	else
	{
		n_master=n;
		masterSet=effectiveGroundSet;
	}
	
	numEffectiveGroundset = effectiveGroundSet.size();
	similarityWithNearestInEffectiveX.resize(n_master, 0);
}

//For sparse mode
FacilityLocation::FacilityLocation(ll n_, std::string mode_, std::vector<float>arr_val, std::vector<ll>arr_count, std::vector<ll>arr_col, ll num_neighbors_, bool partial_, std::set<ll> ground_)
{

	//std::cout<<n_<<" "<<mode_<<" "<<num_neighbors_<<" "<<partial_<<"\n";
	if (mode_ != "sparse") 
	{
		std::cerr << "Error: Incorrect mode specified for the provided sparse similarity matrix\n";
		return;
	}

	if (arr_val.size() == 0 || arr_count.size() == 0 || arr_col.size() == 0)
	{
		std::cerr << "Error: Empty/Corrupt similarity matrix\n";
		return;
	}

	n = n_;
	mode = mode_;
	k_sparse = SparseSim(arr_val, arr_count, arr_col);
	num_neighbors = num_neighbors_;
	partial = partial_;
	seperateMaster = false;
	//Populating effectiveGroundSet
	if (partial == true)
	{
		effectiveGroundSet = ground_;
	}
	else
	{
		for (ll i = 0; i < n; ++i)
		{
			effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
		}
	}
	
	//Populating masterSet
	n_master=n;
	masterSet=effectiveGroundSet;
	
	numEffectiveGroundset = effectiveGroundSet.size();
	similarityWithNearestInEffectiveX.resize(n_master, 0);
	//std::cerr<<"To be implemented\n";
}

//For cluster mode
FacilityLocation::FacilityLocation(ll n_, std::string mode_, std::vector<std::set<ll>>clusters_,std::vector<std::vector<std::vector<float>>>v_k_cluster_, std::vector<ll>v_k_ind_, ll num_neighbors_, bool partial_, std::set<ll> ground_ )
{
	//std::cout<<"A\n";
	/*
	for(int i=0;i<clusters_.size();++i)
	{	
		std::set<ll>ci = clusters_[i];
		for (auto it = ci.begin(); it != ci.end(); ++it)
		{
			std::cout<<*it<<" ";
		}
		std::cout<<"\n";
	}*/

	if (mode_ != "clustered")
	{
		std::cerr << "Error: Incorrect mode specified for the provided cluster\n";
		return;
	}

	if (clusters_.size() == 0)
	{
		std::cerr << "Error: Cluster vector is empty\n";
		return;
	}

	n = n_;
	num_cluster = clusters_.size();
	mode = mode_;
	clusters = clusters_;
	v_k_cluster = v_k_cluster_;
	v_k_ind = v_k_ind_;
	num_neighbors = num_neighbors_;
	partial = partial_;
	seperateMaster = false;
	//Populating ground set
	if (partial == true)
	{
		effectiveGroundSet = ground_;
	}
	else
	{
		for (ll i = 0; i < n; ++i)
		{
			effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
		}
	}

	//Populating masterSet
	n_master=n;
	masterSet=effectiveGroundSet;

	numEffectiveGroundset = effectiveGroundSet.size();
	clusterIDs.resize(n);

	for(int i=0;i<num_cluster;++i)//O(n) (One time operation)
	{
		std::set<ll>ci=clusters[i];
		for (auto it = ci.begin(); it != ci.end(); ++it)
		{
			ll ind = *it;
			clusterIDs[ind]=i;
		}
	}

	relevantX.resize(num_cluster);
	clusteredSimilarityWithNearestInRelevantX.resize(n, 0);
	
	/*for(ll i=0;i<num_cluster;++i)//////////
	{
		std::set<ll>temp;
		relevantX.push_back(temp);
	}*/
}

//helper friend function
float get_max_sim_dense(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj)
{
	ll i = datapoint_ind, j; //i comes from masterSet and j comes from X (which is a subset of groundSet)
	auto it = dataset_ind.begin();
	float m = obj.k_dense[i][*it];

	for (; it != dataset_ind.end(); ++it)//search max similarity wrt datapoints of given dataset
	{
		ll j = *it;
		if (obj.k_dense[i][j] > m)
		{
			m = obj.k_dense[i][j];
		}
	}

	return m;
}
float get_max_sim_sparse(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj)
{
	//std::cout<<"C\n";
	ll i = datapoint_ind, j; //i comes from masterSet and j comes from X (which is a subset of groundSet)
	auto it = dataset_ind.begin();
	float m = obj.k_sparse.get_val(i,*it);
	//std::cout<<m<<"\n";
	
	for (; it != dataset_ind.end(); ++it)//search max similarity wrt datapoints of given dataset
	{
		//std::cout<<"D\n";
		ll j = *it;
		float temp_val = obj.k_sparse.get_val(i,j);
		if (temp_val > m)
		{
			m = temp_val;
		}
	}

	return m;
}

float get_max_sim_cluster(ll datapoint_ind, std::set<ll> dataset_ind, FacilityLocation obj, ll cluster_id)
{

	ll i = datapoint_ind, j, i_, j_; 
	auto it = dataset_ind.begin();
	i_ = obj.v_k_ind[i];
	j_ = obj.v_k_ind[*it]; 
	float m = obj.v_k_cluster[cluster_id][i_][j_];
	//Possibly transform i,j to local kernel index

	for (; it != dataset_ind.end(); ++it)
	{
		ll j = *it;
		//Obtain local kernel indicies for given cluster
		i_ = obj.v_k_ind[i];
		j_ = obj.v_k_ind[j];
		if (obj.v_k_cluster[cluster_id][i_][j_] > m)
		{
			m = obj.v_k_cluster[cluster_id][i_][j_];
		}
	}

	return m;
}


//TODO: In all the methods below, get rid of code redundancy by merging dense and sparse mode blocks together
float FacilityLocation::evaluate(std::set<ll> X)
{
	std::set<ll> effectiveX;
	float result=0;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));

	}
	else
	{
		effectiveX = X;
	}

	if(effectiveX.size()==0)//verify if returning 0 here is correct
	{
		return 0;
	}
	


	if (mode == "dense")
	{
		//Implementing f(X)=Sum_i_V ( max_j_X ( s_ij ) )
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it) //O(n^2) where n=num of elements in effective GS 
		{
			ll ind = *it;
			//std::cout<<ind<<" "<<get_max_sim_dense(ind, effectiveX, *this)<<"\n";
			result += get_max_sim_dense(ind, effectiveX, *this);
		}
	}
	else
	{
		if (mode == "sparse")
		{
			//std::cout<<"A\n";
			for (auto it = masterSet.begin(); it != masterSet.end(); ++it) //O(n^2) where n=num of elements in effective GS 
			{
				ll ind = *it;
				result += get_max_sim_sparse(ind, effectiveX, *this);
			}
		}
		else
		{
			if (mode == "clustered")
			{
				//std::cout<<"A\n";
				for(ll i=0;i<num_cluster;++i)
				{
					//std::cout<<"B\n";
					std::set<ll>releventSubset;
					std::set<ll>ci = clusters[i];
					std::set_intersection(X.begin(), X.end(), ci.begin(), ci.end(), std::inserter(releventSubset, releventSubset.begin()));

					if(releventSubset.size()==0)//if no intersection, skip to next cluster
					{
						continue;
					}
					
					for (auto it = ci.begin(); it != ci.end(); ++it)
					{
						//std::cout<<"C\n";
						ll ind = *it;
						result += get_max_sim_cluster(ind, releventSubset, *this, i);
					}
					
				}
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return result;
}


float FacilityLocation::evaluateWithMemoization(std::set<ll> X) //assumes that pre computed statistics exist for effectiveX
{
	std::set<ll> effectiveX;
	float result = 0;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));

	}
	else
	{
		effectiveX = X;
	}

	if(effectiveX.size()==0)//verify if returning 0 here is correct
	{
		return 0;
	}

	if (mode == "dense")
	{
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
		{
			ll ind = *it;
			result += similarityWithNearestInEffectiveX[ind];
		}
	}
	else
	{
		if (mode == "sparse")
		{
			for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
			{
				ll ind = *it;
				result += similarityWithNearestInEffectiveX[ind];
			}

		}
		else
		{
			if (mode == "clustered")
			{
				for(ll i=0;i<num_cluster;++i)
				{
					std::set<ll>ci = clusters[i];
					if(relevantX[i].size()==0)
					{
						continue;
					}
					
					for (auto it = ci.begin(); it != ci.end(); ++it)
					{
						ll ind = *it;
						result += clusteredSimilarityWithNearestInRelevantX[ind];
					}

				}
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return result;
}


float FacilityLocation::marginalGain(std::set<ll> X, ll item)
{
	std::set<ll> effectiveX;
	float gain = 0;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (effectiveGroundSet.find(item) == effectiveGroundSet.end()) //O(log(n))
	{
		return 0;
	}

	if (effectiveX.find(item) != effectiveX.end())
	{
		return 0;
	}

	if (mode == "dense")
	{
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
		{
			ll ind = *it;
			float m = get_max_sim_dense(ind, effectiveX, *this);
			if (k_dense[ind][item] > m)
			{
				gain += (k_dense[ind][item] - m);
			}
		}
	}
	else
	{
		if (mode == "sparse")
		{
			//std::cout<<"A\n";
			for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
			{
				//std::cout<<"B\n";
				ll ind = *it;
				float m = get_max_sim_sparse(ind, effectiveX, *this);
				float temp_val = k_sparse.get_val(ind,item);
				if (temp_val > m)
				{
					gain += (temp_val - m);
				}
			}

		}
		else
		{
			if (mode == "clustered")
			{
				ll i = clusterIDs[item];
				std::set<ll>releventSubset;
				std::set<ll>ci = clusters[i];
				std::set_intersection(X.begin(), X.end(), ci.begin(), ci.end(), std::inserter(releventSubset, releventSubset.begin()));

				if(releventSubset.size()==0)
				{
					for (auto it = ci.begin(); it != ci.end(); ++it)
					{
						ll ind = *it;
						ll ind_=v_k_ind[ind];
						ll item_ = v_k_ind[item];
						gain+=v_k_cluster[i][ind_][item_];
					}
				}
				else
				{
					for (auto it = ci.begin(); it != ci.end(); ++it)
					{
						ll ind = *it;
						ll ind_=v_k_ind[ind];
						ll item_ = v_k_ind[item];
						float m = get_max_sim_cluster(ind, releventSubset, *this, i);
						if (v_k_cluster[i][ind_][item_] > m)
						{
							gain += (v_k_cluster[i][ind_][item_] - m);
						}
					}
								
				}
				

			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return gain;
}


float FacilityLocation::marginalGainWithMemoization(std::set<ll> X, ll item)
{
	std::set<ll> effectiveX;
	float gain = 0;
	//std::cout<<"G\n";
	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (effectiveGroundSet.find(item) == effectiveGroundSet.end())
	{
		//std::cout<<"J1\n";
		return 0;
	}

	if (effectiveX.find(item) != effectiveX.end())
	{
		//std::cout<<"J2\n";
		return 0;
	}

	if (mode == "dense")
	{
		//std::cout<<"H\n";
		for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
		{
			//std::cout<<"I\n";
			ll ind = *it;
			if (k_dense[ind][item] > similarityWithNearestInEffectiveX[ind])
			{
				gain += (k_dense[ind][item] - similarityWithNearestInEffectiveX[ind]);
			}

		}
	}
	else
	{
		if (mode == "sparse")
		{
			for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
			{
				ll ind = *it;
				float temp_val = k_sparse.get_val(ind,item);
				if (temp_val > similarityWithNearestInEffectiveX[ind])
				{
					gain += (temp_val - similarityWithNearestInEffectiveX[ind]);
				}

			}
		}
		else
		{
			if (mode == "clustered")
			{
				ll i = clusterIDs[item];
				std::set<ll>releventSubset = relevantX[i];
				std::set<ll>ci = clusters[i];
				
				if(releventSubset.size()==0)
				{
					for (auto it = ci.begin(); it != ci.end(); ++it)
					{
						ll ind = *it;
						ll ind_=v_k_ind[ind];
						ll item_ = v_k_ind[item];
						gain+=v_k_cluster[i][ind_][item_];
					}
				}
				else
				{
					for (auto it = ci.begin(); it != ci.end(); ++it)
					{
						ll ind = *it;
						ll ind_=v_k_ind[ind];
						ll item_ = v_k_ind[item];
						float temp_val = v_k_cluster[i][ind_][item_];
						
						if (temp_val > clusteredSimilarityWithNearestInRelevantX[ind])
						{
							gain += (temp_val - clusteredSimilarityWithNearestInRelevantX[ind]);
						}
					}
					
				}
				
			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}

	return gain;
}

void FacilityLocation::updateMemoization(std::set<ll> X, ll item)
{

	//std::cout<<"E\n";
	std::set<ll> effectiveX;

	if (partial == true)
	{
		//effectiveX = intersect(X, effectiveGroundSet)
		std::set_intersection(X.begin(), X.end(), effectiveGroundSet.begin(), effectiveGroundSet.end(), std::inserter(effectiveX, effectiveX.begin()));
	}
	else
	{
		effectiveX = X;
	}

	if (effectiveGroundSet.find(item) == effectiveGroundSet.end())
	{
		return;
	}
	if (X.find(item) != X.end())
	{
		return;
	}

	if (mode == "dense")
	{

		for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
		{
			ll ind = *it;
			//std::cout<<ind<<" "<<item<<"\n";
			if (k_dense[ind][item] > similarityWithNearestInEffectiveX[ind])
			{
				similarityWithNearestInEffectiveX[ind] = k_dense[ind][item];
			}

		}
	}
	else
	{
		if (mode == "sparse")
		{
			for (auto it = masterSet.begin(); it != masterSet.end(); ++it)
			{
				ll ind = *it;
				float temp_val = k_sparse.get_val(ind,item);
				if (temp_val > similarityWithNearestInEffectiveX[ind])
				{
					similarityWithNearestInEffectiveX[ind] = temp_val;
				}

			}
		}
		else
		{
			if (mode == "clustered")
			{
				ll i = clusterIDs[item];
				std::set<ll>ci = clusters[i];
				for (auto it = ci.begin(); it != ci.end(); ++it)
				{
					ll ind = *it;
					ll ind_=v_k_ind[ind];
					ll item_ = v_k_ind[item];
					float temp_val = v_k_cluster[i][ind_][item_];	
					if (temp_val > clusteredSimilarityWithNearestInRelevantX[ind])
					{
						clusteredSimilarityWithNearestInRelevantX[ind]= temp_val;
					}		 
		
				}
				relevantX[i].insert(item);

			}
			else
			{
				std::cerr << "ERROR: INVALID mode\n";
			}
		}

	}
}

std::set<ll> FacilityLocation::getEffectiveGroundSet()
{
	return effectiveGroundSet;
}


std::vector<std::pair<ll, float>> FacilityLocation::maximize(std::string s,float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbosity=false)//TODO: migrate fixed things to constructor
{

	if(s=="NaiveGreedy")
	{
		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbosity);
	} 
}


void FacilityLocation::cluster_init(ll n_, std::vector<std::vector<float>>k_dense_, std::set<ll> ground_)
{
	*this = FacilityLocation(n_, "dense", k_dense_, -1, true, ground_, false);
}

void FacilityLocation::clearMemoization()
{
	//TODO: essentially we want to reset similarityWithNearestInEffectiveX for dense and sparse modes and we want to reset relevantX and clusteredSimilarityWithNearestInRelevantX for clustered mode

	//TODO: Refer https://stackoverflow.com/questions/55266468/whats-the-fastest-way-to-reinitialize-a-vector/55266856 to replace it with a more efficient implementation

	if(mode=="dense" || mode=="sparse")
	{
		for(int i=0;i<n_master;++i)
		{
			similarityWithNearestInEffectiveX[i]=0;
		}
	}
	if(mode == "clustered")
	{
		for(int i=0;i<n;++i)
		{
			clusteredSimilarityWithNearestInRelevantX[i]=0;
		}
	}
		
}

void FacilityLocation::setMemoization(std::set<ll> X) 
{
    clearMemoization();
    std::set<ll>temp;
	for (auto it = X.begin(); it != X.end(); ++it)
	{	
		updateMemoization(temp, *it);
		temp.insert(*it);	
	}
}


