//Note that this wrapper doesn't work properly because in maximize() method, we are trying to bind
//a class argument in an implicit manner which is wrong in pybind11
#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include "../submod/NaiveGreedyOptimizer.h"
#include "wrapper.h"
//#include "../submod/wrapper.h"

namespace py = pybind11;

typedef long long int ll;

void cl_NaiveGreedyOptimizer(py::module &m)
{
py::class_<NaiveGreedyOptimizer>(m,"NaiveGreedyOptimizer")
    //.def(py::init<FacilityLocation>())
    .def(py::init<>())
    .def("maximize", &NaiveGreedyOptimizer::maximize);
}