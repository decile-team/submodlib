#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"Clustered.h"

Clustered::Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>>clusters_, std::vector<std::vector<std::vector<float>>>clusterKernels_, std::vector<ll>clusterIndexMap_ ) {
    // std::cout << "Clustered multi constructor\n";
    n = n_;
    mode = "many_cluster_kernels";
    num_clusters = clusters_.size();
    function_name = function_name_;
    clusters = clusters_;
    clusterKernels = clusterKernels_;
    clusterIndexMap = clusterIndexMap_;
    clusterIDs.resize(n);
    clusters_translated.resize(num_clusters);

    for (ll i = 0; i < n; ++i) {
        effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
    }

    for(int i=0;i<num_clusters;++i) { //O(n) (One time operation)
		std::unordered_set<ll> ci=clusters[i];
		for (auto it = ci.begin(); it != ci.end(); ++it) {
			ll ind = *it;
			clusterIDs[ind]=i;
            clusters_translated[i].insert(clusterIndexMap[ind]);//populating translated indicies
		}
	}

    for(int i=0;i<num_clusters;++i) {
        std::unordered_set<ll> ci = clusters_translated[i];//initilize function object with translated cluster system as that will be consistent with indicies in corresponding kernel
        std::vector<std::vector<float>>kernel = clusterKernels[i];
        SetFunction *f_obj;
        if(function_name=="FacilityLocation") {
            f_obj = new FacilityLocation;
        } else if(function_name == "DisparitySum") {
            f_obj = new DisparitySum;
        }
        f_obj->cluster_init(ci.size(), kernel, ci, false); 
        mixture.push_back(f_obj);
    }
}

Clustered::Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>>clusters_, std::vector<std::vector<float>>denseKernel_) {
    // std::cout << "Clustered single constructor\n";
    n = n_;
    mode = "single_dense_kernel";
    num_clusters = clusters_.size();
    denseKernel = denseKernel_;
    function_name = function_name_;
    clusters = clusters_;
    clusterIDs.resize(n);

    for (ll i = 0; i < n; ++i) {
        effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
    }

    for(int i=0;i<num_clusters;++i) {
        std::unordered_set<ll> ci = clusters[i];
        for (auto it = ci.begin(); it != ci.end(); ++it) {
			ll ind = *it;
			clusterIDs[ind]=i;
		}
        SetFunction *f_obj;
        if(function_name=="FacilityLocation") {
            f_obj = new FacilityLocation;
        } else if(function_name == "DisparitySum") {
            f_obj = new DisparitySum;
        }
        f_obj->cluster_init(n, denseKernel, ci, true); 
        mixture.push_back(f_obj);
    }
}

std::unordered_set<ll> translate_X(std::unordered_set<ll> X, Clustered obj, ll cluster_id) { //Before using X, its important to translate it to suitable form
    std::unordered_set<ll> X_res;
    for (auto it = X.begin(); it != X.end(); ++it) {
        ll ind = *it;
        if(obj.clusterIDs[ind]==cluster_id) { //if given data index is in current cluster then translate it to suitable index and put it in X_res
            X_res.insert(obj.clusterIndexMap[ind]);
        }
    }
    return X_res;
}

float Clustered::evaluate(std::unordered_set<ll> X) {
    // std::cout << "Clustered evaluate\n";
    float res=0;
    if (mode == "single_dense_kernel") {
        for(int i=0;i<num_clusters;++i) {
            res += mixture[i]->evaluate(X);
        }
    } else {
        for(int i=0;i<num_clusters;++i) {
            std::unordered_set<ll> X_temp = translate_X(X, *this, i);
            res+=mixture[i]->evaluate(X_temp);
        }
    }
    return res;
}

float Clustered::evaluateWithMemoization(std::unordered_set<ll> X) {
    // std::cout << "Clustered evaluateWithMemoization\n";
    float res=0;
    if(mode == "single_dense_kernel") {
        for(int i=0;i<num_clusters;++i) {
            res+=mixture[i]->evaluateWithMemoization(X);
        }
    } else {
        for(int i=0;i<num_clusters;++i) {
            std::unordered_set<ll> X_temp = translate_X(X, *this, i);
            res+=mixture[i]->evaluateWithMemoization(X_temp);
        }
    }
    return res;
}

float Clustered::marginalGain(std::unordered_set<ll> X, ll item) {
    // std::cout << "Clustered marginalGain\n";
    ll i = clusterIDs[item];
    if (mode == "single_dense_kernel") {
        return mixture[i]->marginalGain(X, item);
    } else {
        std::unordered_set<ll> X_temp = translate_X(X, *this, i);
        ll item_temp = clusterIndexMap[item];
        
        // if(X_temp.size()==0) {
        //     float gain=0;
        //     std::unordered_set<ll>ci = clusters[i];

        //     for (auto it = ci.begin(); it != ci.end(); ++it) {
        //         ll ind = *it;
        //         ll ind_=clusterIndexMap[ind];
        //         ll item_ = clusterIndexMap[item];
        //         gain+=clusterKernels[i][ind_][item_];
        //     }
        //     return gain;
        // }
        return mixture[i]->marginalGain(X_temp, item_temp);
    }
}

float Clustered::marginalGainWithMemoization(std::unordered_set<ll> X, ll item) {
    // std::cout << "Clustered marginalGainWithMemoization\n";
    ll i = clusterIDs[item];
    if (mode == "single_dense_kernel") {
        return mixture[i]->marginalGainWithMemoization(X, item);
    } else {
        std::unordered_set<ll> X_temp = translate_X(X, *this, i);
        ll item_temp = clusterIndexMap[item];

        // if(X_temp.size()==0) {
        //     float gain=0;
        //     std::unordered_set<ll>ci = clusters[i];

        //     for (auto it = ci.begin(); it != ci.end(); ++it)
        //     {
        //         ll ind = *it;
        //         ll ind_=clusterIndexMap[ind];
        //         ll item_ = clusterIndexMap[item];
        //         gain+=clusterKernels[i][ind_][item_];
        //     }
        //     return gain;
        // }
        return mixture[i]->marginalGainWithMemoization(X_temp, item_temp);
    }
}

void Clustered::updateMemoization(std::unordered_set<ll> X, ll item)
{
    // std::cout << "Clustered updateMemoization\n";
    ll i = clusterIDs[item];
    if (mode == "single_dense_kernel") {
        mixture[i]->updateMemoization(X, item);
    } else {
        std::unordered_set<ll> X_temp = translate_X(X, *this, i);
        ll item_temp = clusterIndexMap[item];
        mixture[i]->updateMemoization(X_temp, item_temp);
    }
}

std::unordered_set<ll> Clustered::getEffectiveGroundSet()
{
    // std::cout << "Clustered getEffectiveGroundSet\n";
    return effectiveGroundSet;
}

std::vector<std::pair<ll, float>> Clustered::maximize(std::string optimizer,float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbosity=false)
{
    // std::cout << "Clustered maximize\n";
	if(optimizer=="NaiveGreedy")
	{
		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbosity);
	} else {
        std::cerr << "Optimizer not yet implemented" << std::endl;
    }
}

void Clustered::clearMemoization() {
    // std::cout << "Clustered clearMemoization\n";
    for(int i=0;i<num_clusters;++i) {
        mixture[i]->clearMemoization();
    }
}  

void Clustered::setMemoization(std::unordered_set<ll> X) {
    // std::cout << "Clustered setMemoization\n";

    if(mode == "single_dense_kernel") {
        for(int i=0;i<num_clusters;++i) {
            mixture[i]->setMemoization(X);
        }
    } else {
        for(int i=0;i<num_clusters;++i) {
            std::unordered_set<ll> X_temp = translate_X(X, *this, i);
            mixture[i]->setMemoization(X_temp);
        }
    }
}
