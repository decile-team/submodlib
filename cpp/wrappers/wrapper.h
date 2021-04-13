#include <unordered_set>

namespace py = pybind11;

void cl_helper(py::module &);
void cl_FacilityLocation(py::module_ &); 
void cl_FeatureBased(py::module_ &); 
void cl_GraphCut(py::module_ &); 
void cl_SetCover(py::module_ &);
void cl_ProbabilisticSetCover(py::module_ &);
void cl_DisparitySum(py::module_ &); 
void cl_LogDeterminant(py::module_ &); 
void cl_DisparityMin(py::module_ &); 
void cl_FacilityLocationMutualInformation(py::module_ &); 
void cl_FacilityLocationVariantMutualInformation(py::module_ &); 
void cl_ConcaveOverModular(py::module_ &); 
void cl_GraphCutMutualInformation(py::module_ &); 
void cl_GraphCutConditionalGain(py::module_ &); 
void cl_sparse_utils(py::module &);
void cl_Clustered(py::module &);
//void cl_set(py::module &);
//void cl_NaiveGreedyOptimizer(py::module &);
//void cl_ClusteredFunction(py::module_ &); 
//void cl_Helper(py::module_ &);
//void fun_NaiveGreedy(py::module_ &);