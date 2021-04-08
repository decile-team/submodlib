#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../smi/FacilityLocationMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_FacilityLocationMutualInformation(py::module &m)
{
    py::class_<FacilityLocationMutualInformation>(m,"FacilityLocationMutualInformation")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, float>()) //dense 
        .def("evaluate", &FacilityLocationMutualInformation::evaluate)
        .def("evaluateWithMemoization", &FacilityLocationMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &FacilityLocationMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &FacilityLocationMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &FacilityLocationMutualInformation::updateMemoization)
        .def("clearMemoization", &FacilityLocationMutualInformation::clearMemoization)
        .def("setMemoization", &FacilityLocationMutualInformation::setMemoization)
        .def("getEffectiveGroundSet", &FacilityLocationMutualInformation::getEffectiveGroundSet)
        .def("maximize", &FacilityLocationMutualInformation::maximize);
}