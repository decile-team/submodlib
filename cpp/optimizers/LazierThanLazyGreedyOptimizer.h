#ifndef LAZIERTHANLAZYGREEDYOPTIMIZER_H
#define LAZIERTHANLAZYGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class LazierThanLazyGreedyOptimizer 
{
    public:
    LazierThanLazyGreedyOptimizer();
    std::vector<std::pair<ll, double>> maximize(SetFunction &f_obj, ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, float epsilon=0.1, bool verbose=false, bool showProgress=true);
    bool equals(double val1, double val2, double eps);
};
#endif


