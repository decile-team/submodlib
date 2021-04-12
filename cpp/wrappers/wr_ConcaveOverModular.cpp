#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../smi/ConcaveOverModular.h"
#include "wrapper.h"
#include<unordered_set>

namespace py = pybind11;

void cl_ConcaveOverModular(py::module &m)
{
    py::class_<ConcaveOverModular> com(m,"ConcaveOverModular");
    com.def(py::init<ll, int, std::vector<std::vector<float>>&, float, ConcaveOverModular::Type>())
        .def("evaluate", &ConcaveOverModular::evaluate)
        .def("evaluateWithMemoization", &ConcaveOverModular::evaluateWithMemoization)
        .def("marginalGain", &ConcaveOverModular::marginalGain)
        .def("marginalGainWithMemoization", &ConcaveOverModular::marginalGainWithMemoization)
        .def("updateMemoization", &ConcaveOverModular::updateMemoization)
        .def("clearMemoization", &ConcaveOverModular::clearMemoization)
        .def("setMemoization", &ConcaveOverModular::setMemoization)
        .def("getEffectiveGroundSet", &ConcaveOverModular::getEffectiveGroundSet)
        .def("maximize", &ConcaveOverModular::maximize);
    
    py::enum_<ConcaveOverModular::Type>(com, "Type")
    .value("squareRoot", ConcaveOverModular::Type::squareRoot)
    .value("inverse", ConcaveOverModular::Type::inverse)
    .value("logarithmic", ConcaveOverModular::Type::logarithmic)
    .export_values();
}