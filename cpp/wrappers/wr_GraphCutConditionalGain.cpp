#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../condgain/GraphCutConditionalGain.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_GraphCutConditionalGain(py::module &m)
{
    py::class_<GraphCutConditionalGain>(m,"GraphCutConditionalGain")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, float, float>()) //dense 
        .def("evaluate", &GraphCutConditionalGain::evaluate)
        .def("evaluateWithMemoization", &GraphCutConditionalGain::evaluateWithMemoization)
        .def("marginalGain", &GraphCutConditionalGain::marginalGain)
        .def("marginalGainWithMemoization", &GraphCutConditionalGain::marginalGainWithMemoization)
        .def("updateMemoization", &GraphCutConditionalGain::updateMemoization)
        .def("clearMemoization", &GraphCutConditionalGain::clearMemoization)
        .def("setMemoization", &GraphCutConditionalGain::setMemoization)
        .def("getEffectiveGroundSet", &GraphCutConditionalGain::getEffectiveGroundSet)
        .def("maximize", &GraphCutConditionalGain::maximize);
}