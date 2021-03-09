#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class NaiveGreedyOptimizer 
{
    public:
    NaiveGreedyOptimizer();
    std::vector<std::pair<ll, float>> maximize(SetFunction &f_obj, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);
    bool equals(double val1, double val2, double eps);
};
#endif


