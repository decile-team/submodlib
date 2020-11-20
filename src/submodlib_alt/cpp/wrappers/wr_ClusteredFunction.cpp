#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include"/submodlib/cpp/ClusteredFunction.h"
#include "wrapper.h"

namespace py = pybind11;


void cl_ClusteredFunction(py::module_ &m)
{
    py::class_<ClusteredFunction>(m,"ClusteredFunction")
        //constructor(no_of_elem_in_ground, vector_of_clusters, function_name, sim_matrix)
        .def(py::init<int, std::vector<std::set<float>>, std::string, std::vector<std::vector<float>>>()) //dense_sim_matrix
        .def(py::init<int, std::vector<std::set<float>>, std::string, std::vector<std::vector<std::pair<int,float>>>>()) //sparse_sim_matrix1 (array of list)
        .def(py::init<int, std::vector<std::set<float>>, std::string, std::pair<std::pair<std::vector<int>, std::vector<int>>>()) //sparse_sim_matrix2 (csr)
        .def("evaluate", &ClusteredFunction::evaluate);
        .def("evaluateSequential", &ClusteredFunction::evaluateSequential);
        .def("marginalGain", &ClusteredFunction::marginalGain);
        .def("marginalGainSequential", &ClusteredFunction::marginalGainSequential);
        .def("sequentialUpdate", &ClusteredFunction::sequentialUpdate);

}