#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../smi/LogDeterminantMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_LogDeterminantMutualInformation(py::module &m)
{
    py::class_<LogDeterminantMutualInformation>(m,"LogDeterminantMutualInformation")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, double, float>()) //dense 
        .def("evaluate", &LogDeterminantMutualInformation::evaluate)
        .def("evaluateWithMemoization", &LogDeterminantMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &LogDeterminantMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &LogDeterminantMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &LogDeterminantMutualInformation::updateMemoization)
        .def("clearMemoization", &LogDeterminantMutualInformation::clearMemoization)
        .def("setMemoization", &LogDeterminantMutualInformation::setMemoization)
        .def("getEffectiveGroundSet", &LogDeterminantMutualInformation::getEffectiveGroundSet)
        .def("maximize", &LogDeterminantMutualInformation::maximize);
}