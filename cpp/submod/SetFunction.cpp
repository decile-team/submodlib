
#include<set>
#include<vector>
#include<utility>
#include<string>
#include"SetFunction.h"
float SetFunction::evaluate(std::set<ll> X){}
float SetFunction::evaluateSequential(std::set<ll> X){}
float SetFunction::marginalGain(std::set<ll> X, ll item){}
float SetFunction::marginalGainSequential(std::set<ll> X, ll item){}
void SetFunction::sequentialUpdate(std::set<ll> X, ll item){}
std::set<ll> SetFunction::getEffectiveGroundSet(){}
std::vector<std::pair<ll, float>> SetFunction::maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity){}
