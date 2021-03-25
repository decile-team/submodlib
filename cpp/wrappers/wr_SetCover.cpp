#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/SetCover.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_SetCover(py::module &m)
{
    py::class_<SetCover>(m, "SetCover")
        .def(py::init<ll, std::vector<std::unordered_set<int>>&, int, std::vector<float> &>())  
        .def("evaluate", &SetCover::evaluate)
        .def("evaluateWithMemoization", &SetCover::evaluateWithMemoization)
        .def("marginalGain", &SetCover::marginalGain)
        .def("marginalGainWithMemoization", &SetCover::marginalGainWithMemoization)
        .def("updateMemoization", &SetCover::updateMemoization)
        .def("getEffectiveGroundSet", &SetCover::getEffectiveGroundSet)
        .def("clearMemoization", &SetCover::clearMemoization)
        .def("setMemoization", &SetCover::setMemoization)
        .def("maximize", &SetCover::maximize);
}