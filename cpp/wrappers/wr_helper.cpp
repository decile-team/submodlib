#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include"../utils/helper.h"
#include "wrapper.h"

namespace py = pybind11;

void cl_helper(py::module &m)
{
    m.def("dot_prod", &dot_prod);
    m.def("mag", &mag);
    m.def("cosine_similarity", &cosine_similarity);
    m.def("euclidean_distance", &euclidean_distance);
    m.def("euclidean_similarity", &euclidean_similarity);
    m.def("create_kernel", &create_kernel);
    m.def("create_kernel_NS", &create_kernel_NS);
    m.def("create_square_kernel_dense", &create_square_kernel_dense);
}