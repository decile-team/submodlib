#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/FeatureBased.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_FeatureBased(py::module &m)
{
    py::class_<FeatureBased> fb(m, "FeatureBased");
    fb.def(py::init<ll, FeatureBased::Type, std::vector<std::vector<std::pair<int, float>>> &, int, std::vector<float> &>()) //dense 
        .def("evaluate", &FeatureBased::evaluate)
        .def("evaluateWithMemoization", &FeatureBased::evaluateWithMemoization)
        .def("marginalGain", &FeatureBased::marginalGain)
        .def("marginalGainWithMemoization", &FeatureBased::marginalGainWithMemoization)
        .def("updateMemoization", &FeatureBased::updateMemoization)
        .def("getEffectiveGroundSet", &FeatureBased::getEffectiveGroundSet)
        .def("clearMemoization", &FeatureBased::clearMemoization)
        .def("setMemoization", &FeatureBased::setMemoization)
        .def("maximize", &FeatureBased::maximize);
    
    py::enum_<FeatureBased::Type>(fb, "Type")
    .value("squareRoot", FeatureBased::Type::squareRoot)
    .value("inverse", FeatureBased::Type::inverse)
    .value("logarithmic", FeatureBased::Type::logarithmic)
    .export_values();
}