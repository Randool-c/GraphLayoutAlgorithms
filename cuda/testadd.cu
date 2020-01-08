#include<cuda_runtime.h>
#include<device_launch_parameters.h>
//#include<helper_cuda.h>
#include<cuda.h>
#include<stdio.h>

#define BLOCK_X 4
#define BLOCK_Y 4
#define N 8


__global__ void add(double *a, double *b, double *c, double *dout){
    int i = threadIdx.x + BLOCK_X * blockIdx.x;
    int j = threadIdx.y + BLOCK_Y * blockIdx.y;
    int idx = i * N + j;
    c[idx] = a[idx] + b[idx];
    __syncthreads();
    c[0] = a[0] + b[0];
    dout[0] = 1919;
    dout[1] = 9199;
}


int main(){
    double *a, *b, *out;
    double *d_a, *d_b, *d_out;
    double *iout, *dout;

    a = (double*)malloc(sizeof(double) * N * N);
    b = (double*)malloc(sizeof(double) * N * N);
    out = (double*)malloc(sizeof(double) * N * N);
    iout = (double*)malloc(sizeof(double) * 2);
    cudaMalloc((void**)&d_out, sizeof(double) * N * N);
    cudaMalloc((void**)&d_a, sizeof(double) * N * N);
    cudaMalloc((void**)&d_b, sizeof(double) * N * N);
    cudaMalloc((void**)&dout, sizeof(double) * 2);

    for (int i = 0 ; i < N * N; ++i){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMemcpy(d_a, a, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(double) * N * N, cudaMemcpyHostToDevice);

    int bx = 1 + (N - 1) / BLOCK_X;
    int by = 1 + (N - 1) / BLOCK_Y;
    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X, BLOCK_Y);

    add<<<dimGrid, dimBlock>>>(d_a, d_b, d_out, dout);

    cudaMemcpy(out, d_out, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(iout, dout, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * N; ++i){
        printf("%f ", out[i]);
    }
    printf("\n");
    printf("%f\n", iout[0]);
    printf("%f\n", iout[1]);
    return 0;
}
