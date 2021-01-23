#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/FacilityLocation.h"
#include "wrapper.h"

namespace py = pybind11;

typedef long long int ll;

void cl_FacilityLocation(py::module &m)
{
    py::class_<FacilityLocation>(m,"FacilityLocation")
        //constructor(no_of_elem_in_ground, mode, sim_matrix or cluster, num_neigh, partial, ground_subset )
        .def(py::init<ll, std::string, std::vector<std::vector<float>>, ll, bool, std::set<ll>, bool >()) //dense matrix
        .def(py::init<ll, std::string, std::vector<float>, std::vector<ll>, std::vector<ll>, ll, bool, std::set<ll>>()) //sparse matrix
        .def(py::init<ll, std::string, std::vector<std::set<ll>>, std::vector<std::vector<std::vector<float>>>, std::vector<ll>, ll, bool, std::set<ll>>()) //cluster
        .def("evaluate", &FacilityLocation::evaluate)
        .def("evaluateWithMemoization", &FacilityLocation::evaluateWithMemoization)
        .def("marginalGain", &FacilityLocation::marginalGain)
        .def("marginalGainWithMemoization", &FacilityLocation::marginalGainWithMemoization)
        .def("updateMemoization", &FacilityLocation::updateMemoization)
        .def("getEffectiveGroundSet", &FacilityLocation::getEffectiveGroundSet)
        .def("clearMemoization", &FacilityLocation::clearMemoization)
        .def("setMemoization", &FacilityLocation::setMemoization)
        .def("maximize", &FacilityLocation::maximize);
        
}