#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../condgain/LogDeterminantConditionalGain.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_LogDeterminantConditionalGain(py::module &m)
{
    py::class_<LogDeterminantConditionalGain>(m,"LogDeterminantConditionalGain")
        .def(py::init<ll, int, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, double, float>()) //dense 
        .def("evaluate", &LogDeterminantConditionalGain::evaluate)
        .def("evaluateWithMemoization", &LogDeterminantConditionalGain::evaluateWithMemoization)
        .def("marginalGain", &LogDeterminantConditionalGain::marginalGain)
        .def("marginalGainWithMemoization", &LogDeterminantConditionalGain::marginalGainWithMemoization)
        .def("updateMemoization", &LogDeterminantConditionalGain::updateMemoization)
        .def("clearMemoization", &LogDeterminantConditionalGain::clearMemoization)
        .def("setMemoization", &LogDeterminantConditionalGain::setMemoization)
        .def("getEffectiveGroundSet", &LogDeterminantConditionalGain::getEffectiveGroundSet)
        .def("maximize", &LogDeterminantConditionalGain::maximize);
}