#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class NaiveGreedyOptimizer 
{
    public:
    NaiveGreedyOptimizer();
    std::vector<std::pair<ll, double>> maximize(SetFunction &f_obj, ll budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbose=false, bool showProgress=true, const std::vector<int>& costs=std::vector<int>(), bool costSensitiveGreedy=false);
    bool equals(double val1, double val2, double eps);
};
#endif


