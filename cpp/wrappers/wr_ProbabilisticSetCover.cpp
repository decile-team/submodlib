#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/ProbabilisticSetCover.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_ProbabilisticSetCover(py::module &m)
{
    py::class_<ProbabilisticSetCover>(m, "ProbabilisticSetCover")
        .def(py::init<ll, std::vector<std::vector<float>>&, int, std::vector<float> &>())  
        .def("evaluate", &ProbabilisticSetCover::evaluate)
        .def("evaluateWithMemoization", &ProbabilisticSetCover::evaluateWithMemoization)
        .def("marginalGain", &ProbabilisticSetCover::marginalGain)
        .def("marginalGainWithMemoization", &ProbabilisticSetCover::marginalGainWithMemoization)
        .def("updateMemoization", &ProbabilisticSetCover::updateMemoization)
        .def("getEffectiveGroundSet", &ProbabilisticSetCover::getEffectiveGroundSet)
        .def("clearMemoization", &ProbabilisticSetCover::clearMemoization)
        .def("setMemoization", &ProbabilisticSetCover::setMemoization)
        .def("maximize", &ProbabilisticSetCover::maximize);
}