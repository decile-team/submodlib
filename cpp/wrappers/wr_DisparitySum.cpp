#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/DisparitySum.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_DisparitySum(py::module &m)
{
    py::class_<DisparitySum>(m,"DisparitySum")
        .def(py::init<ll, std::vector<std::vector<float>>&, bool, std::unordered_set<ll>&>()) //dense 
        .def(py::init<ll, std::vector<float>&, std::vector<ll>&, std::vector<ll>& >()) //sparse 
        .def("evaluate", &DisparitySum::evaluate)
        .def("evaluateWithMemoization", &DisparitySum::evaluateWithMemoization)
        .def("marginalGain", &DisparitySum::marginalGain)
        .def("marginalGainWithMemoization", &DisparitySum::marginalGainWithMemoization)
        .def("updateMemoization", &DisparitySum::updateMemoization)
        .def("getEffectiveGroundSet", &DisparitySum::getEffectiveGroundSet)
        .def("clearMemoization", &DisparitySum::clearMemoization)
        .def("setMemoization", &DisparitySum::setMemoization)
        .def("maximize", &DisparitySum::maximize);
}