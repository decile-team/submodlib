#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../cmi/FacilityLocationConditionalMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_FacilityLocationConditionalMutualInformation(py::module &m)
{
    py::class_<FacilityLocationConditionalMutualInformation>(m,"FacilityLocationConditionalMutualInformation")
        .def(py::init<ll, int, int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, float, float>())  
        .def("evaluate", &FacilityLocationConditionalMutualInformation::evaluate)
        .def("evaluateWithMemoization", &FacilityLocationConditionalMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &FacilityLocationConditionalMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &FacilityLocationConditionalMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &FacilityLocationConditionalMutualInformation::updateMemoization)
        .def("clearMemoization", &FacilityLocationConditionalMutualInformation::clearMemoization)
        .def("setMemoization", &FacilityLocationConditionalMutualInformation::setMemoization)
        .def("getEffectiveGroundSet", &FacilityLocationConditionalMutualInformation::getEffectiveGroundSet)
        .def("maximize", &FacilityLocationConditionalMutualInformation::maximize);
}