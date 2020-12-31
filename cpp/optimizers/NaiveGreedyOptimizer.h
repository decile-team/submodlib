//Note that this code is in prototype state. Its a direct implementation of the pseudo code and there seems to be some logical errors.
#include"optimizers.h"
#include"../submod/SetFunction.h"

typedef long long int ll;

class NaiveGreedyOptimizer : public optimizers
{
    //FacilityLocation f_obj;

    public:
    NaiveGreedyOptimizer();
    //NaiveGreedyOptimizer(FacilityLocation obj_); 
    std::vector<std::pair<ll, float>> maximize(SetFunction &f_obj, float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);
    //std::vector<std::pair<int, float>> maximize(float budget, bool stopIfZeroGain, bool stopIfNegativeGain, bool verbosity);

};
