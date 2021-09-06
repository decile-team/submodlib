#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"Clustered.h"

Clustered::Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>> const&clusters_, std::vector<std::vector<std::vector<float>>> const &clusterKernels_, std::vector<ll> const &clusterIndexMap_, float lambda_): n(n_), mode(multi), num_clusters(clusters_.size()), function_name(function_name_), clusters(clusters_), clusterKernels(clusterKernels_), clusterIndexMap(clusterIndexMap_), lambda(lambda_) {
    clusterIDs.resize(n);
    clusters_translated.resize(num_clusters);
    effectiveGroundSet.reserve(n);
    for (ll i = 0; i < n; ++i) {
        effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
    }

    for(int i=0;i<num_clusters;++i) { 
        //populate clusterIDs and clusters_translated for this cluster
		std::unordered_set<ll> ci=clusters[i];
		//for (auto it = ci.begin(); it != ci.end(); ++it) {
        for (auto ind: ci) {
			//ll ind = *it;
			clusterIDs[ind]=i;
            clusters_translated[i].insert(clusterIndexMap[ind]);//populating translated indicies
		}
        //instantiate the component for this cluster and add to the mixture
        std::unordered_set<ll> cti = clusters_translated[i];//initilize function object with translated cluster system as that will be consistent with indicies in corresponding kernel
        std::vector<std::vector<float>>kernel = clusterKernels[i];
        SetFunction *f_obj;
        if(function_name=="FacilityLocation") {
            f_obj = new FacilityLocation();
            f_obj->cluster_init(cti.size(), kernel, cti, false, 1); 
        } else if(function_name == "DisparitySum") {
            f_obj = new DisparitySum;
            f_obj->cluster_init(cti.size(), kernel, cti, false, 1); 
        } else if(function_name == "DisparityMin") {
            f_obj = new DisparityMin;
            f_obj->cluster_init(cti.size(), kernel, cti, false, 1); 
        } else if(function_name == "LogDeterminant") {
            f_obj = new LogDeterminant;
            f_obj->cluster_init(cti.size(), kernel, cti, false, lambda); 
        } else if(function_name == "GraphCut") {
            f_obj = new GraphCut;
            f_obj->cluster_init(cti.size(), kernel, cti, false, lambda); 
        }
        mixture.push_back(f_obj);
	}
}

Clustered::Clustered(ll n_, std::string function_name_, std::vector<std::unordered_set<ll>> const &clusters_, std::vector<std::vector<float>> const &denseKernel_, float lambda_): n(n_), mode(single), num_clusters(clusters_.size()), denseKernel(denseKernel_), function_name(function_name_), clusters(clusters_), lambda(lambda_) {
    // std::cout << "Clustered single constructor\n";
    //n = n_;
    //mode = "single_dense_kernel";
    //num_clusters = clusters_.size();
    //denseKernel = denseKernel_;
    //function_name = function_name_;
    //clusters = clusters_;
    clusterIDs.resize(n);
    effectiveGroundSet.reserve(n);
    for (ll i = 0; i < n; ++i) {
        effectiveGroundSet.insert(i); //each insert takes O(1) time
    }

    for(int i=0;i<num_clusters;++i) {
        std::unordered_set<ll> ci = clusters[i];
        //for (auto it = ci.begin(); it != ci.end(); ++it) {
        for (auto ind: ci) {
			//ll ind = *it;
			clusterIDs[ind]=i;
		}
        SetFunction *f_obj;
        if(function_name=="FacilityLocation") {
            f_obj = new FacilityLocation;
            f_obj->cluster_init(n, denseKernel, ci, true, 1); 
        } else if(function_name == "DisparitySum") {
            f_obj = new DisparitySum;
            f_obj->cluster_init(n, denseKernel, ci, true, 1); 
        } else if(function_name == "DisparityMin") {
            f_obj = new DisparityMin;
            f_obj->cluster_init(n, denseKernel, ci, true, 1); 
        } else if(function_name == "LogDeterminant") {
            //std::cout << "Creating LogDet object\n";
            f_obj = new LogDeterminant;
            f_obj->cluster_init(n, denseKernel, ci, true, lambda); 
        } else if(function_name == "GraphCut") {
            f_obj = new GraphCut;
            f_obj->cluster_init(n, denseKernel, ci, true, lambda); 
        }
        
        mixture.push_back(f_obj);
    }
}

// Clustered* Clustered::clone() {
//     return NULL;
// }

std::unordered_set<ll> translate_X(std::unordered_set<ll> const &X, Clustered const &obj, ll cluster_id) { //Before using X, its important to translate it to suitable form
    std::unordered_set<ll> X_res;
    //for (auto it = X.begin(); it != X.end(); ++it) {
    for (auto ind: X) {
        //ll ind = *it;
        if(obj.clusterIDs[ind]==cluster_id) { //if given data index is in current cluster then translate it to suitable index and put it in X_res
            X_res.insert(obj.clusterIndexMap[ind]);
        }
    }
    return X_res;
}

double Clustered::evaluate(std::unordered_set<ll> const &X) {
    //std::cout << "Clustered evaluate\n";
    // std::cout << "Set to evaluate: {";
    // for(auto elem: X) {
    //     std::cout << elem << ", ";
    // }
    // std::cout << "}\n";
    double res=0;
    if (mode == single) {
        for(int i=0;i<num_clusters;++i) {
            res += mixture[i]->evaluate(X);
        }
    } else {
        //std::cout << "Eval of multi mode\n";
        for(int i=0;i<num_clusters;++i) {
            //std::cout << "Cluster " << i << "\n";
            std::unordered_set<ll> X_temp = translate_X(X, *this, i);
            // std::cout<<"X_temp = {";
            // for(auto elem: X_temp) {
            //     std::cout << elem << ", ";
            // }
            // std::cout << "}\n";
            //std::cout << "Just before DM eval\n";
            res+=mixture[i]->evaluate(X_temp);
        }
    }
    return res;
}

double Clustered::evaluateWithMemoization(std::unordered_set<ll> const &X) {
    // std::cout << "Clustered evaluateWithMemoization\n";
    double res=0;
    if(mode == single) {
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

double Clustered::marginalGain(std::unordered_set<ll> const &X, ll item) {
    // std::cout << "Clustered marginalGain\n";
    ll i = clusterIDs[item];
    if (mode == single) {
        return mixture[i]->marginalGain(X, item);
    } else {
        std::unordered_set<ll> X_temp = translate_X(X, *this, i);
        ll item_temp = clusterIndexMap[item];
        
        // if(X_temp.size()==0) {
        //     double gain=0;
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

double Clustered::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item, bool enableChecks) {
    // std::cout << "Clustered marginalGainWithMemoization\n";
    ll i = clusterIDs[item];
    if (mode == single) {
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

void Clustered::updateMemoization(std::unordered_set<ll> const &X, ll item)
{
    // std::cout << "Clustered updateMemoization\n";
    ll i = clusterIDs[item];
    if (mode == single) {
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

// std::vector<std::pair<ll, double>> Clustered::maximize(std::string optimizer,ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon = 0.1, bool verbose=false, bool showProgress=true)
// {
//     // std::cout << "Clustered maximize\n";
// 	if(optimizer=="NaiveGreedy")
// 	{
// 		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
// 	} else if (optimizer=="LazyGreedy") { 
//         return LazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbose, showProgress);
//     } else if (optimizer=="StochasticGreedy") { 
//         return StochasticGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
//     } else if (optimizer=="LazierThanLazyGreedy") { 
//         return LazierThanLazyGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, epsilon, verbose, showProgress);
//     } else {
//         std::cerr << "Invalid Optimizer" << std::endl;
//     }
// }

void Clustered::clearMemoization() {
    // std::cout << "Clustered clearMemoization\n";
    for(int i=0;i<num_clusters;++i) {
        mixture[i]->clearMemoization();
    }
}  

void Clustered::setMemoization(std::unordered_set<ll> const &X) {
    // std::cout << "Clustered setMemoization\n";

    if(mode == single) {
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
