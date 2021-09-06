#ifndef FLCG_H
#define FLCG_H
#include "ConditionalGain.h"
#include "../submod/FacilityLocation.h"

class FacilityLocationConditionalGain : public SetFunction {
   protected:
    ll n;
    int numPrivates;
    float privacyHardness;
    std::unordered_set<ll> indexCorrectedP;
    std::vector<std::vector<float>> kernelImage;       // n X n
    std::vector<std::vector<float>> kernelPrivate;       // n X numPrivates
    std::vector<std::vector<float>> superKernel;       //n X n+numPrivates
    FacilityLocation *facLoc;
    ConditionalGain *condGain;

   public:
    FacilityLocationConditionalGain(
        ll n_, int numPrivates_,
        std::vector<std::vector<float>> const &kernelImage_,
        std::vector<std::vector<float>> const &kernelPrivate_,
        float privacyHardness_);

    double evaluate(std::unordered_set<ll> const &X);
    double evaluateWithMemoization(std::unordered_set<ll> const &X);
    double marginalGain(std::unordered_set<ll> const &X, ll item);
    double marginalGainWithMemoization(std::unordered_set<ll> const &X,
                                       ll item, bool enableChecks=true);
    void updateMemoization(std::unordered_set<ll> const &X, ll item);
    std::unordered_set<ll> getEffectiveGroundSet();
    // std::vector<std::pair<ll, double>> maximize(std::string, ll budget,
    //                                             bool stopIfZeroGain,
    //                                             bool stopIfNegativeGain,
    //                                             float epsilon, bool verbose, bool showProgress);
    void clearMemoization();
    void setMemoization(std::unordered_set<ll> const &X);
    // FacilityLocationConditionalGain* clone();
};
#endif