#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>
#include "compute_stress.h"

namespace py = pybind11;


int add(int i, int j){
    return i + j;
}


//std::vector<double>& f(std::vector<double> &a){
//    double *n = new double[5];
//    memcpy(n, &a[0], sizeof(double) * 5);
//    for (int i = 0; i < 5; ++i){
//        printf("%f\n", n[i]);
//    }
////    vector<double> ans(n);
//    return std::vector<double> ans(n, n + 5);
//}

PYBIND11_MODULE(compute_stress, m){
    m.doc() = "pybind11";
//    m.def("add", &add, "pybind11 example ");
//    m.def("f", &f, "print first element");
//    m.def("f", [](std::vector<double> &a){
//        return std::vector<double>(a);
//    });
    m.def("compute_stress_cpu", &compute_stress_cpu, "compute stress on cpu");
    m.def("compute_stress_gpu", &compute_stress_gpu, "compute stress on gpu");
}
