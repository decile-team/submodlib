#include <unordered_set>

namespace py = pybind11;

void cl_helper(py::module &);
void cl_FacilityLocation(py::module_ &); 
void cl_FeatureBased(py::module_ &); 
void cl_GraphCut(py::module_ &); 
void cl_DisparitySum(py::module_ &); 
void cl_sparse_utils(py::module &);
void cl_Clustered(py::module &);
//void cl_set(py::module &);
//void cl_NaiveGreedyOptimizer(py::module &);
//void cl_ClusteredFunction(py::module_ &); 
//void cl_Helper(py::module_ &);
//void fun_NaiveGreedy(py::module_ &);