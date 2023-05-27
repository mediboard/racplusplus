#include <pybind11/pybind11.h>
#include "racplusplus.h"

PYBIND11_MODULE(racplusplus, handle){
    m.doc() = "doc( 
        RACplusplus is a C++ optimized python package for performing
        reciprocal agglomerative clustering.

        Authors: Porter Hunley, Daniel Frees
        2023
    )doc"

    m.def("rac", &RAC, R"fdoc(
        Run RAC algorithm on a provided array of points.

        Returns an array of the group # each point was assigned to.
    )fdoc");

    m.def("test_rac", &main, R"fdoc(
        Testing function to run and time RAC's run in C++.
    )fdoc");

    m.attr("__version__") = "0.9";
}
