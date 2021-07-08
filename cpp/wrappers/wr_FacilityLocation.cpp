#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/FacilityLocation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_FacilityLocation(py::module &m)
{
    py::class_<FacilityLocation>(m,"FacilityLocation")
        .def(py::init<ll, std::vector<std::vector<float>>&, bool, std::unordered_set<ll>&, bool >()) //dense 
        .def(py::init<ll, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, bool, std::string >()) //dense 
        .def(py::init<ll, std::vector<float>&, std::vector<ll>&, std::vector<ll>& >()) //sparse 
        .def(py::init<ll, std::vector<std::unordered_set<ll>>&, std::vector<std::vector<std::vector<float>>>&, std::vector<ll>& >()) //cluster
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