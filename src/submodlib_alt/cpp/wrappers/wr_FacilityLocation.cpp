#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"/submodlib/cpp/submod/FacilityLocation.h"
#include "wrapper.h"

namespace py = pybind11;


void cl_FacilityLocation(py::module_ &m)
{
    py::class_<FacilityLocation>(m,"FacilityLocation")
        //constructor(no_of_elem_in_ground, mode, sim_matrix, num_neigh, cluster, partial, subset)
        .def(py::init<int, std::string, std::vector<std::vector<float>>, int, std::set<float>, bool, std::set<float>>()) //dense_sim_matrix
        .def(py::init<int, std::string, std::vector<std::vector<std::pair<int,float>>>, int, std::set<float>, bool, std::set<float>>()) //sparse_sim_matrix1 (array of list)
        .def(py::init<int, std::string, std::pair<std::pair<std::vector<int>, std::vector<int>>, std::vector<float>>, int, std::set<float>, bool, std::set<float>>()) //sparse_sim_matrix2 (csr)
        .def("evaluate", &FacilityLocation::evaluate);
        .def("evaluateSequential", &FacilityLocation::evaluateSequential);
        .def("marginalGain", &FacilityLocation::marginalGain);
        .def("marginalGainSequential", &FacilityLocation::marginalGainSequential);
        .def("sequentialUpdate", &FacilityLocation::sequentialUpdate);
        .def("getEffectiveGroundSet", &FacilityLocation::getEffectiveGroundSet);


}