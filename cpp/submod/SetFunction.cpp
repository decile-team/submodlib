
#include<set>
#include<vector>
#include<utility>
#include<string>
#include"SetFunction.h"
float SetFunction::evaluate(std::set<ll> X){}
float SetFunction::evaluateWithMemoization(std::set<ll> X){}
float SetFunction::marginalGain(std::set<ll> X, ll item){}
float SetFunction::marginalGainWithMemoization(std::set<ll> X, ll item){}
void SetFunction::updateMemoization(std::set<ll> X, ll item){}
std::set<ll> SetFunction::getEffectiveGroundSet(){}
std::vector<std::pair<ll, float>> SetFunction::maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity){}
void SetFunction::cluster_init(ll n_, std::vector<std::vector<float>>k_dense_, std::set<ll> ground_){}
void SetFunction::setMemoization(std::set<ll> X){}
void SetFunction::clearPreCompute(){}