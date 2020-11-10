#include <pybind11/pybind11.h>
#include "HelloWorld.h"
namespace py = pybind11;

PYBIND11_MODULE(submodlib_cpp, m) {
	py::class_<mess>(m, "mess")
		.def(py::init<std::string>())
		.def("out", &mess::out);
}
