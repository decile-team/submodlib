#ifndef NAIVEGREEDYOPTIMIZER_H
#define NAIVEGREEDYOPTIMIZER_H
#include"../SetFunction.h"
#include <unordered_set>

class NaiveGreedyOptimizer 
{
    public:
    NaiveGreedyOptimizer();
    std::vector<std::pair<ll, double>> maximize(SetFunction &f_obj, float budget, bool stopIfZeroGain=false, bool stopIfNegativeGain=false, bool verbose=false, bool showProgress=true, const std::vector<float>& costs=std::vector<float>(), bool costSensitiveGreedy=false);
    bool equals(double val1, double val2, double eps);
};
#endif


