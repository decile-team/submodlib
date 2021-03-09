#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/DisparitySum.h"
#include "wrapper.h"

namespace py = pybind11;

typedef long long int ll;

void cl_DisparitySum(py::module &m)
{
    py::class_<DisparitySum>(m,"DisparitySum")
        .def(py::init<ll, std::string, std::vector<std::vector<float>>, ll, bool, std::set<ll> >()) //dense matrix
        .def(py::init<ll, std::string, std::vector<float>, std::vector<ll>, std::vector<ll>, ll, bool, std::set<ll>>()) //sparse matrix
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