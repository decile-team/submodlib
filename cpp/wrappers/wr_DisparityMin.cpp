#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/DisparityMin.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_DisparityMin(py::module &m)
{
    py::class_<DisparityMin>(m,"DisparityMin")
        .def(py::init<ll, std::vector<std::vector<float>>&, bool, std::unordered_set<ll>&>()) //dense 
        .def(py::init<ll, std::vector<float>&, std::vector<ll>&, std::vector<ll>& >()) //sparse 
        .def("evaluate", &DisparityMin::evaluate)
        .def("evaluateWithMemoization", &DisparityMin::evaluateWithMemoization)
        .def("marginalGain", &DisparityMin::marginalGain)
        .def("marginalGainWithMemoization", &DisparityMin::marginalGainWithMemoization)
        .def("updateMemoization", &DisparityMin::updateMemoization)
        .def("getEffectiveGroundSet", &DisparityMin::getEffectiveGroundSet)
        .def("clearMemoization", &DisparityMin::clearMemoization)
        .def("setMemoization", &DisparityMin::setMemoization)
        .def("maximize", &DisparityMin::maximize);
}