#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/LogDeterminant.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_LogDeterminant(py::module &m)
{
    py::class_<LogDeterminant>(m,"LogDeterminant")
        .def(py::init<ll, std::vector<std::vector<float>>&, bool, std::unordered_set<ll>&, double>()) //dense 
        .def(py::init<ll, std::vector<float>&, std::vector<ll>&, std::vector<ll>&, double >()) //sparse 
        .def("evaluate", &LogDeterminant::evaluate)
        .def("evaluateWithMemoization", &LogDeterminant::evaluateWithMemoization)
        .def("marginalGain", &LogDeterminant::marginalGain)
        .def("marginalGainWithMemoization", &LogDeterminant::marginalGainWithMemoization)
        .def("updateMemoization", &LogDeterminant::updateMemoization)
        .def("getEffectiveGroundSet", &LogDeterminant::getEffectiveGroundSet)
        .def("clearMemoization", &LogDeterminant::clearMemoization)
        .def("setMemoization", &LogDeterminant::setMemoization)
        .def("maximize", &LogDeterminant::maximize);
}