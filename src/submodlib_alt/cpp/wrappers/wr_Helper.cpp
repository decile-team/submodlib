#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"/submodlib/cpp/Helper.h"
#include "wrapper.h"

namespace py = pybind11;


void cl_Helper(py::module_ &m)
{
    py::class_<Helper>(m,"Helper")
        .def(py::init<>())
        .def_static("createDenseKernel", &Helper::createDenseKernel) 
        .def_static("createSparseKernel", &Helper::createSparseKernel)
        .def_static("createClusters", &Helper::createClusters)  

}