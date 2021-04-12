#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../smi/GraphCutMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_GraphCutMutualInformation(py::module &m)
{
    py::class_<GraphCutMutualInformation>(m,"GraphCutMutualInformation")
        .def(py::init<ll, int, std::vector<std::vector<float>>&>()) //dense 
        .def("evaluate", &GraphCutMutualInformation::evaluate)
        .def("evaluateWithMemoization", &GraphCutMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &GraphCutMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &GraphCutMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &GraphCutMutualInformation::updateMemoization)
        .def("clearMemoization", &GraphCutMutualInformation::clearMemoization)
        .def("setMemoization", &GraphCutMutualInformation::setMemoization)
        .def("getEffectiveGroundSet", &GraphCutMutualInformation::getEffectiveGroundSet)
        .def("maximize", &GraphCutMutualInformation::maximize);
}