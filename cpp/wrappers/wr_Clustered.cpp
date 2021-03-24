#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../Clustered.h"
#include<unordered_set>
#include "wrapper.h"

namespace py = pybind11;

void cl_Clustered(py::module &m)
{
    py::class_<Clustered>(m,"Clustered")
        .def(py::init<ll, std::string , std::vector<std::unordered_set<ll>>&, std::vector<std::vector<std::vector<float>>>&, std::vector<ll>&, float>())
        .def(py::init<ll, std::string , std::vector<std::unordered_set<ll>>&, std::vector<std::vector<float>>&, float>()) 
        .def("evaluate", &Clustered::evaluate)
        .def("evaluateWithMemoization", &Clustered::evaluateWithMemoization)
        .def("marginalGain", &Clustered::marginalGain)
        .def("marginalGainWithMemoization", &Clustered::marginalGainWithMemoization)
        .def("updateMemoization", &Clustered::updateMemoization)
        .def("getEffectiveGroundSet", &Clustered::getEffectiveGroundSet)
        .def("clearMemoization", &Clustered::clearMemoization)
        .def("setMemoization", &Clustered::setMemoization)
        .def("maximize", &Clustered::maximize);
}