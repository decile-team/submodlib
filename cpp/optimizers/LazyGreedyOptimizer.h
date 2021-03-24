#ifndef LAZYGREEDYOPTIMIZER_H
#define LAZYGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class LazyGreedyOptimizer 
{
    public:
    LazyGreedyOptimizer();
    std::vector<std::pair<ll, double>> maximize(SetFunction &f_obj, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbose);
    bool equals(double val1, double val2, double eps);
};
#endif


