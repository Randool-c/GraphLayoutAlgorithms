#include "a.h"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
//#include<helper_cuda.h>
#include<cuda.h>
#define BLOCK_X 4
#define BLOCK_Y 4
#define N 8

__global__ void add_(double *a, double *b, double *c){
    int i = threadIdx.x + BLOCK_X * blockIdx.x;
    int j = threadIdx.y + BLOCK_Y * blockIdx.y;
    int idx = i * N + j;
    c[idx] = a[idx] + b[idx];
}

ClassA::ClassA(int size) {
    n = size;
    cudaMalloc((void**)&d_data, sizeof(double) * size);
}

ClassA::ClassA(ClassA &&other) {
    n = other.n;
    cudaMalloc((void**)&d_data, sizeof(double) * n);
    cudaMemcpy(d_data, other.d_data, sizeof(double) * n, cudaMemcpyDeviceToDevice);
}

ClassA ClassA::add(ClassA &other) {
    ClassA ans(N * N);
    int bx = 1 + (N - 1) / BLOCK_X;
    int by = 1 + (N - 1) / BLOCK_Y;
    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    add_<<<dimGrid, dimBlock>>>(d_data, other.d_data, ans.d_data);
    return ans;
}

ClassA::~ClassA() {
    cudaFree(d_data);
}

void ClassA::to_host() {
    data = (double*)malloc(sizeof(double) * n);
    cudaMemcpy(data, d_data, sizeof(double) * n, cudaMemcpyDeviceToHost);
}

ClassA random(int n){
    ClassA ans(n);
    double *x = (double*)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i){
        x[i] = rand() % 10;
        cout << x[i] << endl;
    }
    cudaMemcpy(ans.d_data, x, sizeof(double) * n, cudaMemcpyHostToDevice);
    return ans;
}
