#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/GraphCut.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_GraphCut(py::module &m)
{
    py::class_<GraphCut>(m,"GraphCut")
        .def(py::init<ll, std::vector<std::vector<float>>&, bool, std::unordered_set<ll>&, float >()) //dense 
        .def(py::init<ll, std::vector<std::vector<float>>&, std::vector<std::vector<float>>&, float >()) //dense 
        .def(py::init<ll, std::vector<float>&, std::vector<ll>&, std::vector<ll>&, float >()) //sparse 
        .def("evaluate", &GraphCut::evaluate)
        .def("evaluateWithMemoization", &GraphCut::evaluateWithMemoization)
        .def("marginalGain", &GraphCut::marginalGain)
        .def("marginalGainWithMemoization", &GraphCut::marginalGainWithMemoization)
        .def("updateMemoization", &GraphCut::updateMemoization)
        .def("getEffectiveGroundSet", &GraphCut::getEffectiveGroundSet)
        .def("clearMemoization", &GraphCut::clearMemoization)
        .def("setMemoization", &GraphCut::setMemoization)
        .def("maximize", &GraphCut::maximize);
}