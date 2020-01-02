//
// Created by chenst on 2020/1/2.
//

#include<cuda.h>
#include<cuda_runtime.h>
// #include<device_launch_parameters.h>
#include<cstdio>
#include<iostream>
using namespace std;


__host__ __device__ int run_on_cpu_or_gpu(){
    return 1;
}

__global__ void run_on_gpu(){
    printf("run on gpu %d\n", run_on_cpu_or_gpu());
}

int main(){
    // cudaMemcpy
    int deviceCount;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    cout << "name: " << deviceProp.name << endl;
    cout << "compute capablity: " << deviceProp.major << '.' << deviceProp.minor << endl;
    
    printf("run on cpu %d\n", run_on_cpu_or_gpu());
    run_on_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
