#include <pybind11/pybind11.h>
#include "wrapper.h"
namespace py = pybind11;

PYBIND11_MODULE(submodlib_cpp, m) 
{
    cl_FacilityLocation(m);
    cl_FacilityLocation2(m);
    cl_FeatureBased(m);
    cl_DisparitySum(m);
    cl_DisparityMin(m);
    cl_GraphCut(m);
    cl_SetCover(m);
    cl_LogDeterminant(m);
    cl_ProbabilisticSetCover(m);

    cl_FacilityLocationMutualInformation(m);
    cl_FacilityLocationVariantMutualInformation(m);
    cl_ConcaveOverModular(m);
    cl_GraphCutMutualInformation(m);
    cl_LogDeterminantMutualInformation(m);
    cl_ProbabilisticSetCoverMutualInformation(m);
    cl_SetCoverMutualInformation(m);

    cl_GraphCutConditionalGain(m);
    cl_FacilityLocationConditionalGain(m);
    cl_LogDeterminantConditionalGain(m);
    cl_ProbabilisticSetCoverConditionalGain(m);
    cl_SetCoverConditionalGain(m);

    cl_FacilityLocationConditionalMutualInformation(m);
    cl_LogDeterminantConditionalMutualInformation(m);
    cl_SetCoverConditionalMutualInformation(m);
    cl_ProbabilisticSetCoverConditionalMutualInformation(m);

    cl_helper(m);
    cl_sparse_utils(m);
    cl_Clustered(m);
    //cl_set(m);
    //cl_NaiveGreedyOptimizer(m);
}
