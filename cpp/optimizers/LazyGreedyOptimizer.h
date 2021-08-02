#ifndef LAZYGREEDYOPTIMIZER_H
#define LAZYGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class LazyGreedyOptimizer 
{
    public:
    LazyGreedyOptimizer();
    std::vector<std::pair<ll, double>> maximize(SetFunction &f_obj, ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbose=false, bool showProgress=true, const std::vector<int>& costs=std::vector<int>(), bool costSensitiveGreedy=false);
    bool equals(double val1, double val2, double eps);
};
#endif


