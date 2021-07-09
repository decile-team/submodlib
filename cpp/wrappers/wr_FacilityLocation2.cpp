#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include <pybind11/numpy.h>
#include"../submod/FacilityLocation2.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_FacilityLocation2(py::module &m)
{
    py::class_<FacilityLocation2>(m,"FacilityLocation2")
        // .def(py::init<ll, py::array_t<float>&, bool, std::unordered_set<ll>&, bool >()) //dense 
        .def(py::init<ll, std::vector<std::vector<float>>&, bool, std::unordered_set<ll>&, bool >()) //dense 
        .def(py::init<ll, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, bool, std::string >()) //dense 
        .def(py::init<ll, std::vector<float>&, std::vector<ll>&, std::vector<ll>& >()) //sparse 
        .def(py::init<ll, std::vector<std::unordered_set<ll>>&, std::vector<std::vector<std::vector<float>>>&, std::vector<ll>& >()) //cluster
        .def(py::init<>())
        .def("pybind_init", &FacilityLocation2::pybind_init)
        .def("evaluate", &FacilityLocation2::evaluate)
        .def("evaluateWithMemoization", &FacilityLocation2::evaluateWithMemoization)
        .def("marginalGain", &FacilityLocation2::marginalGain)
        .def("marginalGainWithMemoization", &FacilityLocation2::marginalGainWithMemoization)
        .def("updateMemoization", &FacilityLocation2::updateMemoization)
        .def("getEffectiveGroundSet", &FacilityLocation2::getEffectiveGroundSet)
        .def("clearMemoization", &FacilityLocation2::clearMemoization)
        .def("setMemoization", &FacilityLocation2::setMemoization)
        .def("maximize", &FacilityLocation2::maximize);
}