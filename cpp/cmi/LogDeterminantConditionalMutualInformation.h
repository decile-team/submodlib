#ifndef LOGDETCMI_H
#define LOGDETCMI_H
#include "../smi/MutualInformation.h"
#include "../condgain/ConditionalGain.h"
#include "../submod/LogDeterminant.h"


class LogDeterminantConditionalMutualInformation : public SetFunction {
   protected:
    ll n;
    int numQueries;
    int numPrivates;
    double lambda;
    float magnificationLambda;
    float privacyHardness;
    std::unordered_set<ll> indexCorrectedQ;
    std::unordered_set<ll> indexCorrectedP;
    std::vector<std::vector<float>> kernelImage;       // n X n
    std::vector<std::vector<float>> kernelQuery;       // n X numQueries
    std::vector<std::vector<float>> kernelQueryQuery;  // numQueries X
                                                       // numQueries
    std::vector<std::vector<float>> kernelPrivate;       // n X numPrivates
    std::vector<std::vector<float>> kernelPrivatePrivate; // numPrivates X numPrivates
    std::vector<std::vector <float> > kernelQueryPrivate;
    std::vector<std::vector<float>> superKernel;       //n+numQueries X n+numQueries
    LogDeterminant *logDet;
    ConditionalGain *condGain;
    MutualInformation *mutualInfo;

   public:
    LogDeterminantConditionalMutualInformation(
        ll n_, int numQueries_, int numPrivates_,
        std::vector<std::vector<float>> const &kernelImage_,
        std::vector<std::vector<float>> const &kernelQuery_,
        std::vector<std::vector<float>> const &kernelQueryQuery_,
        std::vector<std::vector<float>> const &kernelPrivate_,
        std::vector<std::vector<float>> const &kernelPrivatePrivate_,
        std::vector<std::vector<float>> const &kernelQueryPrivate_,
        double lambda_, float magnificationLambda_, float privacyHardness_);

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
    // LogDeterminantConditionalMutualInformation* clone();
};
#endif