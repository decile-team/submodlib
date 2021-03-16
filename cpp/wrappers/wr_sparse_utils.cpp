//This wrapper has been written so that unit testing of sparse_utils can 
//be done using pytest. Apart from that, there is no need to expose sparse_utils
//to Python

#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../utils/sparse_utils.h"
#include "wrapper.h"

namespace py = pybind11;

void cl_sparse_utils(py::module &m)
{
py::class_<SparseSim>(m,"SparseSim")
    .def(py::init<std::vector<float>&, std::vector<ll>&, std::vector<ll>&>())
    .def(py::init<std::vector<float>&, std::vector<ll>&, ll, ll>())
    .def("get_val", &SparseSim::get_val)
    .def("get_row", &SparseSim::get_row)
    .def("get_col", &SparseSim::get_col);
}