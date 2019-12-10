//
// Created by chenst on 2019/12/10.
//

#include "compute_lap_l.h"
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>

namespace py = pybind11;

PYBIND11_MODULE(compute_lap_z, m){
    m.doc() = "pybind11";
    //    m.def("add", &add, "pybind11 example ");
    //    m.def("f", &f, "print first element");
    //    m.def("f", [](std::vector<double> &a){
    //        return std::vector<double>(a);
    //    });
    m.def("compute_lap_z_cpu", [](std::vector<float> &z, std::vector<float> &delta, int n){
        float *result = construct_lap_z_cpu(z, delta, n);
        std::vector<float> ans(result, result + n * n);
        free(result);
        return ans;
    }, "construct z laplacian on cpu");
    m.def("compute_lap_z_gpu", [](std::vector<float> &z, std::vector<float> &delta, int n){
        float *result = construct_lap_z_gpu(z, delta, n);
        std::vector<float> ans(result, result + n * n);
        free(result);
        return ans;
    }, "construct z laplacian on gpu");
}
