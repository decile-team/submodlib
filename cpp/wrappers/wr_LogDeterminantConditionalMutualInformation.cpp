#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../cmi/LogDeterminantConditionalMutualInformation.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_LogDeterminantConditionalMutualInformation(py::module &m)
{
    py::class_<LogDeterminantConditionalMutualInformation>(m,"LogDeterminantConditionalMutualInformation")
        .def(py::init<ll, int, int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, double, float, float>()) //dense 
        .def("evaluate", &LogDeterminantConditionalMutualInformation::evaluate)
        .def("evaluateWithMemoization", &LogDeterminantConditionalMutualInformation::evaluateWithMemoization)
        .def("marginalGain", &LogDeterminantConditionalMutualInformation::marginalGain)
        .def("marginalGainWithMemoization", &LogDeterminantConditionalMutualInformation::marginalGainWithMemoization)
        .def("updateMemoization", &LogDeterminantConditionalMutualInformation::updateMemoization)
        .def("clearMemoization", &LogDeterminantConditionalMutualInformation::clearMemoization)
        .def("setMemoization", &LogDeterminantConditionalMutualInformation::setMemoization)
        .def("getEffectiveGroundSet", &LogDeterminantConditionalMutualInformation::getEffectiveGroundSet)
        .def("maximize", &LogDeterminantConditionalMutualInformation::maximize);
}