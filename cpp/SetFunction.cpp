
#include<set>
#include<vector>
#include<utility>
#include<string>
#include"SetFunction.h"
double SetFunction::evaluate(std::unordered_set<ll> const &X){}
double SetFunction::evaluateWithMemoization(std::unordered_set<ll> const &X){}
double SetFunction::marginalGain(std::unordered_set<ll> const &X, ll item){}
double SetFunction::marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item){}
void SetFunction::updateMemoization(std::unordered_set<ll> const &X, ll item){}
std::unordered_set<ll> SetFunction::getEffectiveGroundSet(){}
std::vector<std::pair<ll, double>> SetFunction::maximize(std::string, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbose){}
void SetFunction::cluster_init(ll n_, std::vector<std::vector<float>> const &k_dense_, std::unordered_set<ll> const &ground_, bool partial, float lambda){}
void SetFunction::setMemoization(std::unordered_set<ll> const &X){}
void SetFunction::clearMemoization(){}
SetFunction * SetFunction::clone() {return NULL;}