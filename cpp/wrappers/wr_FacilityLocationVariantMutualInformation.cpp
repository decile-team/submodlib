#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../smi/FacilityLocationVariantMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_FacilityLocationVariantMutualInformation(py::module &m)
{
    py::class_<FacilityLocationVariantMutualInformation>(m,"FacilityLocationVariantMutualInformation")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, float>()) //dense 
        .def("evaluate", &FacilityLocationVariantMutualInformation::evaluate)
        .def("evaluateWithMemoization", &FacilityLocationVariantMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &FacilityLocationVariantMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &FacilityLocationVariantMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &FacilityLocationVariantMutualInformation::updateMemoization)
        .def("clearMemoization", &FacilityLocationVariantMutualInformation::clearMemoization)
        .def("setMemoization", &FacilityLocationVariantMutualInformation::setMemoization)
        .def("getEffectiveGroundSet", &FacilityLocationVariantMutualInformation::getEffectiveGroundSet)
        .def("maximize", &FacilityLocationVariantMutualInformation::maximize);
}