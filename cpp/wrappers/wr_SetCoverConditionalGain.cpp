#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../condgain/SetCoverConditionalGain.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_SetCoverConditionalGain(py::module &m)
{
    py::class_<SetCoverConditionalGain>(m, "SetCoverConditionalGain")
        .def(py::init<ll, std::vector<std::unordered_set<int>>&, int, std::vector<float> &, std::unordered_set<int> &>())  
        .def("evaluate", &SetCoverConditionalGain::evaluate)
        .def("evaluateWithMemoization", &SetCoverConditionalGain::evaluateWithMemoization)
        .def("marginalGain", &SetCoverConditionalGain::marginalGain)
        .def("marginalGainWithMemoization", &SetCoverConditionalGain::marginalGainWithMemoization)
        .def("updateMemoization", &SetCoverConditionalGain::updateMemoization)
        .def("getEffectiveGroundSet", &SetCoverConditionalGain::getEffectiveGroundSet)
        .def("clearMemoization", &SetCoverConditionalGain::clearMemoization)
        .def("setMemoization", &SetCoverConditionalGain::setMemoization)
        .def("maximize", &SetCoverConditionalGain::maximize);
}