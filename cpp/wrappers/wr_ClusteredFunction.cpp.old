#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/ClusteredFunction.h"
#include "wrapper.h"

namespace py = pybind11;

typedef long long int ll;

void cl_ClusteredFunction(py::module &m)
{
    py::class_<ClusteredFunction>(m,"ClusteredFunction")
        .def(py::init<ll, std::string , std::vector<std::set<ll>>, std::vector<std::vector<std::vector<float>>>, std::vector<ll>>()) 
        .def("evaluate", &ClusteredFunction::evaluate)
        .def("evaluateWithMemoization", &ClusteredFunction::evaluateWithMemoization)
        .def("marginalGain", &ClusteredFunction::marginalGain)
        .def("marginalGainWithMemoization", &ClusteredFunction::marginalGainWithMemoization)
        .def("updateMemoization", &ClusteredFunction::updateMemoization)
        .def("getEffectiveGroundSet", &ClusteredFunction::getEffectiveGroundSet)
        .def("maximize", &ClusteredFunction::maximize);
        
}