#ifndef MI_H
#define MI_H

#include "../optimizers/LazierThanLazyGreedyOptimizer.h"
#include "../optimizers/LazyGreedyOptimizer.h"
#include "../optimizers/NaiveGreedyOptimizer.h"
#include "../optimizers/StochasticGreedyOptimizer.h"
#include "../SetFunction.h"

class MutualInformation : public SetFunction {
   protected:
    ll n;
    SetFunction& f;
    double val_fQ;
    SetFunction *fAUQ;
    std::unordered_set<ll> querySet;
    std::unordered_set<ll> unionPreComputeSet;

   public:
    MutualInformation(SetFunction& f_, std::unordered_set<ll> querySet_);

    double evaluate(std::unordered_set<ll> const &X);
    double evaluateWithMemoization(std::unordered_set<ll> const &X);
    double marginalGain(std::unordered_set<ll> const &X, ll item);
    double marginalGainWithMemoization(std::unordered_set<ll> const &X,
                                       ll item, bool enableChecks=true);
    void updateMemoization(std::unordered_set<ll> const &X, ll item);
    std::unordered_set<ll> getEffectiveGroundSet();
    std::vector<std::pair<ll, double>> maximize(std::string, ll budget,
                                                bool stopIfZeroGain,
                                                bool stopIfNegativeGain,
                                                float epsilon, bool verbose, bool showProgress);
    void clearMemoization();
    void setMemoization(std::unordered_set<ll> const &X);
    virtual ~MutualInformation();
    // MutualInformation* clone();
};

#endif
