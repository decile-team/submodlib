#ifndef LOGDETCG_H
#define LOGDETCG_H
#include "ConditionalGain.h"
#include "../submod/LogDeterminant.h"

class LogDeterminantConditionalGain : public SetFunction {
   protected:
    ll n;
    int numPrivates;
    double lambda;
    float privacyHardness;
    std::unordered_set<ll> indexCorrectedP;
    std::vector<std::vector<float>> kernelImage;       // n X n
    std::vector<std::vector<float>> kernelPrivate;       // n X numPrivates
    std::vector<std::vector<float>> kernelPrivatePrivate; // numPrivates X numPrivates
    std::vector<std::vector<float>> superKernel;       //n X n+numPrivates
    LogDeterminant *logDet;
    ConditionalGain *condGain;

   public:
    LogDeterminantConditionalGain(
        ll n_, int numPrivates_,
        std::vector<std::vector<float>> const &kernelImage_,
        std::vector<std::vector<float>> const &kernelPrivate_,
        std::vector<std::vector<float>> const &kernelPrivatePrivate_,
        double lambda_,
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
    // LogDeterminantConditionalGain* clone();
};
#endif