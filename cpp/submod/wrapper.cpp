#include <pybind11/pybind11.h>
#include<pybind11/stl.h>
#include "wrapper.h"
namespace py = pybind11;

typedef long long int ll;

PYBIND11_MODULE(submodlib_cpp, m) 
{
    cl_FacilityLocation(m);
 //   cl_ClusteredFunction(m);
 //   cl_Helper(m);
 //   fun_NaiveGreedy(m);

}
