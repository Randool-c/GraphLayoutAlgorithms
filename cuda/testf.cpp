//
// Created by chenst on 2019/12/9.
//

#include<stdio.h>
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include<pybind11/numpy.h>
#include<vector>
#include<memory.h>
//#include"compute_stress.h"

namespace py = pybind11;


int add(int i, int j){
    return i + j;
}


void* f(std::vector<double> &a){
    double *n = new double[5];
    memcpy(n, &a[0], sizeof(double) * 5);
    for (int i = 0; i < 5; ++i){
        printf("%f\n", n[i]);
    }
//    vector<double> ans(n);
//    return std::vector<double> ans(n, n + 5);
    return (void*)n;
}

PYBIND11_MODULE(testf, m){
    m.doc() = "pybind11";
    m.def("add", &add, "pybind11 example ");
    m.def("f", &f, "print first element");
//    m.def("f", [](std::vector<double> &a){
//        return std::vector<double>(a);
//    });
}
