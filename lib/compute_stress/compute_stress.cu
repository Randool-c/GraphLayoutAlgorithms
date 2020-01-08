
#include"compute_stress.h"
//#include<Python.h>
//#include<helper_cuda.h>


#define BLOCK_X 32
#define BLOCK_Y 32


__global__ void compute_stress_gpu_kernel(float *pos, float *dist, float *out, int n_nodes){
    int i = threadIdx.x + blockIdx.x * BLOCK_X;
    int j = threadIdx.y + blockIdx.y * BLOCK_Y;
    int idxx = threadIdx.x;
    int idxy = threadIdx.y;

    __shared__ float shared_nums[BLOCK_X][BLOCK_Y];

    if (i == j || i >= n_nodes || j >= n_nodes){
        shared_nums[idxx][idxy] = 0;
    }
    else{
        float d = sqrt((pos[2 * i] - pos[2 * j]) * (pos[2 * i] - pos[2 * j])
                        + (pos[2 * i + 1] - pos[2 * j + 1]) * (pos[2 * i + 1] - pos[2 * j + 1]));
        shared_nums[idxx][idxy] = (d / dist[i * n_nodes + j] - 1);
        shared_nums[idxx][idxy] *= shared_nums[idxx][idxy];
    }

    // reduction on Y
    for (unsigned int stride = BLOCK_Y / 2; stride > 0; stride /= 2){
        __syncthreads();
        if (idxy < stride && idxy + stride < n_nodes){
            shared_nums[idxx][idxy] += shared_nums[idxx][idxy + stride];
        }
    }

    // reduction on X
    if (idxy == 0) {
        for (unsigned int stride = BLOCK_X / 2; stride > 0; stride /= 2) {
            __syncthreads();
            if (idxx < stride && idxx + stride < n_nodes) {
                shared_nums[idxx][0] += shared_nums[idxx + stride][0];
            }
        }
    }

    // atomic sum
    __syncthreads();
    if (idxx == 0 && idxy == 0){
        atomicAdd(out, shared_nums[0][0]);
    }
}

float compute_stress_cpu(std::vector<float> &pos, std::vector<float> &dist, int n){
    int i, j;
    float d;
    float ans = 0.0f;
    for (i = 0; i < n; ++i){
        for (j = 0; j < n; ++j){
            if (i == j){
                continue;
            }
            d = sqrt((pos[2 * i] - pos[2 * j]) * (pos[2 * i] - pos[2 * j])
                            + (pos[2 * i + 1] - pos[2 * j + 1]) * (pos[2 * i + 1] - pos[2 * j + 1]));
            d = (d / dist[i * n + j] - 1);
            d *= d;
            ans += d;
        }
    }
    return ans;
}


float compute_stress_gpu(std::vector<float> &pos, std::vector<float> &dist, int n){
    float *d_pos, *d_dist, *d_out;
    float out;

    // allocate memory on GPU
    cudaMalloc((void**)&d_pos, sizeof(float) * n * 2);
    cudaMalloc((void**)&d_dist, sizeof(float) * n * n);
    cudaMalloc((void**)&d_out, sizeof(float));

    // copy data from cpu to gpu
    cudaMemcpy(d_pos, &pos[0], sizeof(float) * n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, &dist[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int bx = 1 + (n - 1) / BLOCK_X;
    int by = 1 + (n - 1) / BLOCK_Y;
    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    compute_stress_gpu_kernel<<<dimGrid, dimBlock>>>(d_pos, d_dist, d_out, n);

    // copy result from device to host
    cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // free gpu space
    cudaFree(d_pos);
    cudaFree(d_dist);
    cudaFree(d_out);

    cudaDeviceReset();

    return out;
}

//
//int f(int n){
//    double *pos, *dist, out;
//    double *d_pos, *d_dist, *d_out;
//
//    pos = (double*)malloc(sizeof(double) * n * 2);
//    dist = (double*)malloc(sizeof(double) * n * n);
//    cudaMalloc((void**)&d_pos, sizeof(double) * n * 2);
//    cudaMalloc((void**)&d_dist, sizeof(double) * n * n);
//    cudaMalloc((void**)&d_out, sizeof(double));
//
//    // initialize
//    for (int i = 0; i < n; ++i){
//        pos[i * 2] = rand() / double(RAND_MAX);
//        pos[i * 2 + 1] = rand() / (double)(RAND_MAX);
//    }
//    for (int i = 0; i < n * n; ++i){
//        dist[i] = double(rand() % 10) + 1;
//    }
//
//    cudaMemcpy(d_pos, pos, sizeof(double) * n * 2, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_dist, dist, sizeof(double) * n * n, cudaMemcpyHostToDevice);
//    cudaMemset(d_out, 0, sizeof(double));
//
//    int bx = 1 + (n - 1) / BLOCK_X;
//    int by = 1 + (n - 1) / BLOCK_Y;
//    dim3 dimGrid(bx, by);
//    dim3 dimBlock(BLOCK_X, BLOCK_Y);
//    compute_stress_gpu_kernel<<<dimGrid, dimBlock>>>(d_pos, d_dist, d_out, n);
//
//    cudaMemcpy(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
//    double normal_ans = compute_stress(pos, dist, n);
//    printf("%f\n", out);
//    printf("%f\n", normal_ans);
//
//    cudaFree(d_out);
//    cudaFree(d_pos);
//    cudaFree(d_dist);
//    free(pos);
//    free(dist);
//
//    cudaDeviceReset();
//
//    return 0;
//}
