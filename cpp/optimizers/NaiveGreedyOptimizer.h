#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class NaiveGreedyOptimizer 
{
    public:
    NaiveGreedyOptimizer();
    std::vector<std::pair<ll, float>> maximize(SetFunction &f_obj, ll budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbose);
    bool equals(double val1, double val2, double eps);
};
#endif


