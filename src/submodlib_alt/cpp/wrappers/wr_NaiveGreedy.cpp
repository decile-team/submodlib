#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"/submodlib/cpp/optimizers/NaiveGreedy.h"
#include "wrapper.h"

namespace py = pybind11;


void fun_NaiveGreedy(py::module_ &m)
{
    m.def("naiveGreedyMax", &naiveGreedyMax);
}