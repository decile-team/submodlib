#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../condgain/ProbabilisticSetCoverConditionalGain.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_ProbabilisticSetCoverConditionalGain(py::module &m)
{
    py::class_<ProbabilisticSetCoverConditionalGain>(m, "ProbabilisticSetCoverConditionalGain")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<float> &, std::unordered_set<int>&>())  
        .def("evaluate", &ProbabilisticSetCoverConditionalGain::evaluate)
        .def("evaluateWithMemoization", &ProbabilisticSetCoverConditionalGain::evaluateWithMemoization)
        .def("marginalGain", &ProbabilisticSetCoverConditionalGain::marginalGain)
        .def("marginalGainWithMemoization", &ProbabilisticSetCoverConditionalGain::marginalGainWithMemoization)
        .def("updateMemoization", &ProbabilisticSetCoverConditionalGain::updateMemoization)
        .def("getEffectiveGroundSet", &ProbabilisticSetCoverConditionalGain::getEffectiveGroundSet)
        .def("clearMemoization", &ProbabilisticSetCoverConditionalGain::clearMemoization)
        .def("setMemoization", &ProbabilisticSetCoverConditionalGain::setMemoization)
        .def("maximize", &ProbabilisticSetCoverConditionalGain::maximize);
}