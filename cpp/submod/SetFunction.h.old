typedef long long int ll;
class SetFunction
{
    public:
    virtual float evaluate(std::set<ll> X);
    virtual float evaluateWithMemoization(std::set<ll> X);
    virtual float marginalGain(std::set<ll> X, ll item);
    virtual float marginalGainWithMemoization(std::set<ll> X, ll item);
    virtual void updateMemoization(std::set<ll> X, ll item);
    virtual std::set<ll> getEffectiveGroundSet();
    virtual std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);
    virtual void cluster_init(ll n_, std::vector<std::vector<float>>k_dense_, std::set<ll> ground_);
    void setMemoization(std::set<ll> X);
    virtual void clearMemoization();
};