typedef long long int ll;
class SetFunction
{
    public:
    virtual float evaluate(std::set<ll> X);
    virtual float evaluateSequential(std::set<ll> X);
    virtual float marginalGain(std::set<ll> X, ll item);
    virtual float marginalGainSequential(std::set<ll> X, ll item);
    virtual void sequentialUpdate(std::set<ll> X, ll item);
    virtual std::set<ll> getEffectiveGroundSet();
    virtual std::vector<std::pair<ll, float>> maximize(std::string, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);

};