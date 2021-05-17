#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../smi/SetCoverMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_SetCoverMutualInformation(py::module &m)
{
    py::class_<SetCoverMutualInformation>(m, "SetCoverMutualInformation")
        .def(py::init<ll, std::vector<std::unordered_set<int>>&, int, std::vector<float> &, std::unordered_set<int> &>())  
        .def("evaluate", &SetCoverMutualInformation::evaluate)
        .def("evaluateWithMemoization", &SetCoverMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &SetCoverMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &SetCoverMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &SetCoverMutualInformation::updateMemoization)
        .def("getEffectiveGroundSet", &SetCoverMutualInformation::getEffectiveGroundSet)
        .def("clearMemoization", &SetCoverMutualInformation::clearMemoization)
        .def("setMemoization", &SetCoverMutualInformation::setMemoization)
        .def("maximize", &SetCoverMutualInformation::maximize);
}