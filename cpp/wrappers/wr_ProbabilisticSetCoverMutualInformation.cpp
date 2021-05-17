#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../smi/ProbabilisticSetCoverMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_ProbabilisticSetCoverMutualInformation(py::module &m)
{
    py::class_<ProbabilisticSetCoverMutualInformation>(m, "ProbabilisticSetCoverMutualInformation")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<float> &, std::unordered_set<int>&>())  
        .def("evaluate", &ProbabilisticSetCoverMutualInformation::evaluate)
        .def("evaluateWithMemoization", &ProbabilisticSetCoverMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &ProbabilisticSetCoverMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &ProbabilisticSetCoverMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &ProbabilisticSetCoverMutualInformation::updateMemoization)
        .def("getEffectiveGroundSet", &ProbabilisticSetCoverMutualInformation::getEffectiveGroundSet)
        .def("clearMemoization", &ProbabilisticSetCoverMutualInformation::clearMemoization)
        .def("setMemoization", &ProbabilisticSetCoverMutualInformation::setMemoization)
        .def("maximize", &ProbabilisticSetCoverMutualInformation::maximize);
}