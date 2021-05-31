#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../cmi/ProbabilisticSetCoverConditionalMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_ProbabilisticSetCoverConditionalMutualInformation(py::module &m)
{
    py::class_<ProbabilisticSetCoverConditionalMutualInformation>(m, "ProbabilisticSetCoverConditionalMutualInformation")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<float> &, std::unordered_set<int>&, std::unordered_set<int>&>())  
        .def("evaluate", &ProbabilisticSetCoverConditionalMutualInformation::evaluate)
        .def("evaluateWithMemoization", &ProbabilisticSetCoverConditionalMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &ProbabilisticSetCoverConditionalMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &ProbabilisticSetCoverConditionalMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &ProbabilisticSetCoverConditionalMutualInformation::updateMemoization)
        .def("getEffectiveGroundSet", &ProbabilisticSetCoverConditionalMutualInformation::getEffectiveGroundSet)
        .def("clearMemoization", &ProbabilisticSetCoverConditionalMutualInformation::clearMemoization)
        .def("setMemoization", &ProbabilisticSetCoverConditionalMutualInformation::setMemoization)
        .def("maximize", &ProbabilisticSetCoverConditionalMutualInformation::maximize);
}