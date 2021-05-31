#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../cmi/SetCoverConditionalMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_SetCoverConditionalMutualInformation(py::module &m)
{
    py::class_<SetCoverConditionalMutualInformation>(m, "SetCoverConditionalMutualInformation")
        .def(py::init<ll, std::vector<std::unordered_set<int>>&, int, std::vector<float> &, std::unordered_set<int> &, std::unordered_set<int> &>())  
        .def("evaluate", &SetCoverConditionalMutualInformation::evaluate)
        .def("evaluateWithMemoization", &SetCoverConditionalMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &SetCoverConditionalMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &SetCoverConditionalMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &SetCoverConditionalMutualInformation::updateMemoization)
        .def("getEffectiveGroundSet", &SetCoverConditionalMutualInformation::getEffectiveGroundSet)
        .def("clearMemoization", &SetCoverConditionalMutualInformation::clearMemoization)
        .def("setMemoization", &SetCoverConditionalMutualInformation::setMemoization)
        .def("maximize", &SetCoverConditionalMutualInformation::maximize);
}