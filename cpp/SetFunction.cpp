
#include<set>
#include<vector>
#include<utility>
#include<string>
#include"SetFunction.h"
float SetFunction::evaluate(std::unordered_set<ll> const &X){}
float SetFunction::evaluateWithMemoization(std::unordered_set<ll> const &X){}
float SetFunction::marginalGain(std::unordered_set<ll> const &X, ll item){}
float SetFunction::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item){}
void SetFunction::updateMemoization(std::unordered_set<ll> const &X, ll item){}
std::unordered_set<ll> SetFunction::getEffectiveGroundSet(){}
std::vector<std::pair<ll, float>> SetFunction::maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity){}
void SetFunction::cluster_init(ll n_, std::vector<std::vector<float>> const &k_dense_, std::unordered_set<ll> const &ground_, bool partial){}
void SetFunction::setMemoization(std::unordered_set<ll> const &X){}
void SetFunction::clearMemoization(){}
