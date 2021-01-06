#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cmath>
#include<set>
#include<iterator>
#include<map>
#include"ClusteredFunction.h"

ClusteredFunction::ClusteredFunction(ll n_, std::string fun_name_, std::vector<std::set<ll>>clusters_, std::vector<std::vector<std::vector<float>>>v_k_cluster_, std::vector<ll>v_k_ind_ )
{
    n = n_;
    num_cluster = clusters_.size();
    fun_name = fun_name_;
    clusters = clusters_;
    v_k_cluster = v_k_cluster_;
    v_k_ind = v_k_ind_;
    clusterIDs.resize(n);
    clusters_translated.resize(num_cluster);

    for (ll i = 0; i < n; ++i)
    {
        effectiveGroundSet.insert(i); //each insert takes O(log(n)) time
    }

    for(int i=0;i<num_cluster;++i)//O(n) (One time operation)
	{   
		std::set<ll>ci=clusters[i];
		for (auto it = ci.begin(); it != ci.end(); ++it)
		{
			ll ind = *it;
			clusterIDs[ind]=i;
            //std::cout<<v_k_ind[ind]<<" ";
            clusters_translated[i].insert(v_k_ind[ind]);//populating translated indicies
		}
	}

    for(int i=0;i<num_cluster;++i)
    {
        std::set<ll>ci = clusters_translated[i];//initilize function object with translated cluster system as that will be consistent with indicies in corrosponding kernel
        std::vector<std::vector<float>>kernel = v_k_cluster[i];
        //int f_ID = fun_ID[fun_name];
        //SetFunction *f_obj = fun_name_point[f_ID]; 
        SetFunction *f_obj;
        if(fun_name=="FacilityLocation")
        {
            f_obj = new FacilityLocation;
        } else if(fun_name == "DisparitySum") {
            f_obj = new DisparitySum;
        }
        //std::cout<<i<<"\n";
        f_obj->cluster_init(ci.size(),kernel,ci); 
        v_fun.push_back(f_obj);
    }

}

std::set<ll> translate_X(std::set<ll>X, ClusteredFunction obj, ll cluster_id)//Before using X, its important to translate it to suitable form
{
    std::set<ll>X_res;
    //std::cout<<"C\n";
    for (auto it = X.begin(); it != X.end(); ++it)
    {
        ///std::cout<<"D\n";
        ll ind = *it;
        //std::cout<<ind<<" "<<obj.clusterIDs[ind]<<" "<<cluster_id<<"\n";
        if(obj.clusterIDs[ind]==cluster_id)//if given data index is in current cluster then translate it to suitable index and put it in X_res
        {
            X_res.insert(obj.v_k_ind[ind]);
        }
    }

    return X_res;
}



float ClusteredFunction::evaluate(std::set<ll> X)
{
    float res=0;
    //std::cout<<"A\n";
    for(int i=0;i<num_cluster;++i)
    {
        //std::cout<<"B\n";
        std::set<ll>X_temp = translate_X(X, *this, i);
        res+=v_fun[i]->evaluate(X_temp);
    }
    return res;
}


float ClusteredFunction::evaluateSequential(std::set<ll> X)
{
    float res=0;
    for(int i=0;i<num_cluster;++i)
    {
        std::set<ll>X_temp = translate_X(X, *this, i);
        res+=v_fun[i]->evaluateSequential(X);
    }
    return res;
}

float ClusteredFunction::marginalGain(std::set<ll> X, ll item)
{
    ll i = clusterIDs[item];
    std::set<ll>X_temp = translate_X(X, *this, i);
    ll item_temp = v_k_ind[item];
    return v_fun[i]->marginalGain(X_temp, item_temp);
}

float ClusteredFunction::marginalGainSequential(std::set<ll> X, ll item)
{
    ll i = clusterIDs[item];
    std::set<ll>X_temp = translate_X(X, *this, i);
    ll item_temp = v_k_ind[item];
    return v_fun[i]->marginalGainSequential(X_temp, item_temp);
}

void ClusteredFunction::sequentialUpdate(std::set<ll> X, ll item)
{
    ll i = clusterIDs[item];
    std::set<ll>X_temp = translate_X(X, *this, i);
    ll item_temp = v_k_ind[item];
    v_fun[i]->sequentialUpdate(X_temp, item_temp);
}

std::set<ll> ClusteredFunction::getEffectiveGroundSet()
{
    return effectiveGroundSet;
}

std::vector<std::pair<ll, float>> ClusteredFunction::maximize(std::string s,float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbosity=false)
{
    //std::cout<<"A\n";
	if(s=="NaiveGreedy")
	{
		return NaiveGreedyOptimizer().maximize(*this, budget, stopIfZeroGain, stopIfNegativeGain, verbosity);
	} else {
        std::cerr << "Not yet implemented" << std::endl;
    }
}