#ifndef LOGDETMI_H
#define LOGDETMI_H
#include "MutualInformation.h"
#include "../submod/LogDeterminant.h"

class LogDeterminantMutualInformation : public SetFunction {
   protected:
    ll n;
    int numQueries;
    double lambda;
    float magnificationLambda;
    std::unordered_set<ll> indexCorrectedQ;
    std::vector<std::vector<float>> kernelImage;       // n X n
    std::vector<std::vector<float>> kernelQuery;       // n X numQueries
    std::vector<std::vector<float>> kernelQueryQuery;  // numQueries X
                                                       // numQueries
    std::vector<std::vector<float>> superKernel;       //n+numQueries X n+numQueries
    LogDeterminant *logDet;
    MutualInformation *mutualInfo;

   public:
    LogDeterminantMutualInformation(
        ll n_, int numQueries_,
        std::vector<std::vector<float>> const &kernelImage_,
        std::vector<std::vector<float>> const &kernelQuery_,
        std::vector<std::vector<float>> const &kernelQueryQuery_,
        double lambda_, float magnificationLambda_);

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
    virtual ~LogDeterminantMutualInformation();
    // LogDeterminantMutualInformation* clone();
};
#endif
