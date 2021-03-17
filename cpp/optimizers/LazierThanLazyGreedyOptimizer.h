#ifndef LAZIERTHANLAZYGREEDYOPTIMIZER_H
#define LAZIERTHANLAZYGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class LazierThanLazyGreedyOptimizer 
{
    public:
    LazierThanLazyGreedyOptimizer();
    std::vector<std::pair<ll, float>> maximize(SetFunction &f_obj, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, float epsilon, bool verbose);
    bool equals(double val1, double val2, double eps);
};
#endif


