//
// Created by chenst on 2019/12/9.
//

#include<pybind11/pybind11.h>
#include"compute_stress.h"

namespace py = pybind11;

int add(int i, int j){
    return i + j;
}


PYBIND11_MODULE(testf, m){
    m.doc() = "pybind11";
    m.def("add", &add, "pybind11 example ");
    m.def("f", &f, "cuda compute stress");
}
