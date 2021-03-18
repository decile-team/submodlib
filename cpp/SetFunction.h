#ifndef SET_FUNCTION_H
#define SET_FUNCTION_H
#include <unordered_set>
typedef long long int ll;
class SetFunction
{
    public:
    virtual float evaluate(std::unordered_set<ll> const &X);
    virtual float evaluateWithMemoization(std::unordered_set<ll> const &X);
    virtual float marginalGain(std::unordered_set<ll> const &X, ll item);
    virtual float marginalGainWithMemoization(std::unordered_set<ll> const &X, ll item);
    virtual void updateMemoization(std::unordered_set<ll> const &X, ll item);
    virtual std::unordered_set<ll> getEffectiveGroundSet();
    virtual std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbose);
    virtual void cluster_init(ll n_, std::vector<std::vector<float>>const &k_dense_, std::unordered_set<ll> const &ground_, bool partial);
    virtual void setMemoization(std::unordered_set<ll> const &X);
    virtual void clearMemoization();
};
#endif