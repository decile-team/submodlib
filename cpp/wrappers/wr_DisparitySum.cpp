#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../submod/DisparitySum.h"
#include "wrapper.h"

namespace py = pybind11;

typedef long long int ll;

void cl_DisparitySum(py::module &m)
{
    py::class_<DisparitySum>(m,"DisparitySum")
        //constructor(no_of_elem_in_ground, mode, sim_matrix or cluster, num_neigh, partial, ground_subset )
        .def(py::init<ll, std::string, std::vector<std::vector<float>>, ll, bool, std::set<ll> >()) //dense matrix
        .def(py::init<ll, std::string, std::vector<float>, std::vector<ll>, std::vector<ll>, ll, bool, std::set<ll>>()) //sparse matrix
        .def("evaluate", &DisparitySum::evaluate)
        .def("evaluateSequential", &DisparitySum::evaluateSequential)
        .def("marginalGain", &DisparitySum::marginalGain)
        .def("marginalGainSequential", &DisparitySum::marginalGainSequential)
        .def("sequentialUpdate", &DisparitySum::sequentialUpdate)
        .def("getEffectiveGroundSet", &DisparitySum::getEffectiveGroundSet)
        .def("maximize", &DisparitySum::maximize);
}