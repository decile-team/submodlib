#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../condgain/FacilityLocationConditionalGain.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_FacilityLocationConditionalGain(py::module &m)
{
    py::class_<FacilityLocationConditionalGain>(m,"FacilityLocationConditionalGain")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, float>()) //dense 
        .def("evaluate", &FacilityLocationConditionalGain::evaluate)
        .def("evaluateWithMemoization", &FacilityLocationConditionalGain::evaluateWithMemoization)
        .def("marginalGain", &FacilityLocationConditionalGain::marginalGain)
        .def("marginalGainWithMemoization", &FacilityLocationConditionalGain::marginalGainWithMemoization)
        .def("updateMemoization", &FacilityLocationConditionalGain::updateMemoization)
        .def("clearMemoization", &FacilityLocationConditionalGain::clearMemoization)
        .def("setMemoization", &FacilityLocationConditionalGain::setMemoization)
        .def("getEffectiveGroundSet", &FacilityLocationConditionalGain::getEffectiveGroundSet)
        .def("maximize", &FacilityLocationConditionalGain::maximize);
}