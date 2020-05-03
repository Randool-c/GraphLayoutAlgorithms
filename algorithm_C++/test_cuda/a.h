//
// Created by chenst on 2020/4/16.
//

#ifndef MULTILEVEL_STRESS_C___A_H
#define MULTILEVEL_STRESS_C___A_H

//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
////#include<helper_cuda.h>
//#include<cuda.h>
#include<stdio.h>
#include<iostream>
#include<cstdlib>
//#define BLOCK_X 4
//#define BLOCK_Y 4
//#define N 8
using namespace std;

class ClassA{
public:
    int n;
    double *data;
    double *d_data;
    ClassA(int size);
    ClassA(ClassA &&other);
    ClassA add(ClassA &other);
    void to_host();
    ~ClassA();
};

ClassA random(int n);

#endif //MULTILEVEL_STRESS_C___A_H
