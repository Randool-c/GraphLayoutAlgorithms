#include "compute_lap_l.h"

#define BLOCK_X 2
#define BLOCK_Y 2


float *construct_lap_z_cpu(std::vector<float> &z, std::vector<float> &delta, int n){
    float *lap_z = (float*)malloc(sizeof(float) * n * n);
    int i, j, idx;
    float sum_row;
    float d;
    for (i = 0; i < n; ++i){
        sum_row = 0;
        for (j = 0; j < n; ++j){
            idx = i * n + j;
            if (i == j) continue;
            else{
                d = sqrt((z[2 * i] - z[2 * j]) * (z[2 * i] - z[2 * j])
                         + (z[2 * i + 1] - z[2 * j + 1]) * (z[2 * i + 1] - z[2 * j + 1]));
                lap_z[idx] = -delta[idx] / d;
                sum_row += lap_z[idx];
            }
        }
        lap_z[i * n + i] = -sum_row;
    }
    return lap_z;
}


__global__ void construct_lap_z_gpu_kernel(float *z, float *delta, float *out, int n){
    int i = threadIdx.x + blockIdx.x * BLOCK_X;
    int j = threadIdx.y + blockIdx.y * BLOCK_Y;

    int idxx = threadIdx.x;
    int idxy = threadIdx.y;
    int idx = i * n + j;

    __shared__ float row_sum_block[BLOCK_X][BLOCK_Y];

    float d;
    if (i != j && i < n && j < n) {
        d = sqrt((z[2 * i] - z[2 * j]) * (z[2 * i] - z[2 * j])
                 + (z[2 * i + 1] - z[2 * j + 1]) * (z[2 * i + 1] - z[2 * j + 1]));
        out[idx] = -delta[idx] / d;
        row_sum_block[idxx][idxy] = -out[idx];
    }
    else{
        row_sum_block[idxx][idxy] = 0;
    }

    // reduction on each row in the shared matrix
    for (unsigned int stride = BLOCK_X / 2; stride > 0; stride /= 2){
        __syncthreads();
        if (idxx < stride){
            row_sum_block[idxx][idxy] += row_sum_block[idxx + stride][idxy];
        }
    }

    if (idxx == 0 && j < n){
//        printf("%f block idx: (%d, %d); thraed idx: (%d, %d)\n", row_sum_block[0][idxy], blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
        atomicAdd(&out[j * n + j], row_sum_block[0][idxy]);
    }
}


float *construct_lap_z_gpu(std::vector<float> &z, std::vector<float> &delta, int n){
    float *d_z, *d_delta, *d_out;
    float *lap_z;

    lap_z = (float*)malloc(sizeof(float) * n * n);

    // allocate memory on GPU
    cudaMalloc((void**)&d_z, sizeof(float) * n * 2);
    cudaMalloc((void**)&d_delta, sizeof(float) * n * n);
    cudaMalloc((void**)&d_out, sizeof(float) * n * n);

    // copy data from host to device
    cudaMemcpy(d_z, &z[0], sizeof(float) * n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_delta, &delta[0], sizeof(float) * n * n, cudaMemcpyHostToDevice);

    int bx = 1 + (n - 1) / BLOCK_X;
    int by = 1 + (n - 1) / BLOCK_Y;
    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    construct_lap_z_gpu_kernel<<<dimGrid, dimBlock>>>(d_z, d_delta, d_out, n);

    // copy result from device to host
    cudaMemcpy(lap_z, d_out, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

    // free gpu space
    cudaFree(d_z);
    cudaFree(d_delta);
    cudaFree(d_out);
    return lap_z;
}

//void print(float *m, int n){
//    int i, j;
//    for (i = 0; i < n; ++i){
//        for (j = 0; j < n; ++j){
//            printf("%f ", m[i * n + j]);
//        }
//        printf("\n");
//    }
//    printf("\n");
//}
//
//int main(){
//    int n = 10101;
//
//    float *pos = (float*)malloc(sizeof(float) * n * 2);
//    float *delta = (float*)malloc(sizeof(float) * n * n);
//
//
//    for (int i = 0; i < n * 2; ++i){
//        pos[i] = rand() / float(RAND_MAX);
//    }
//    for (int i = 0; i < n; ++i){
//        for (int j = 0; j < n; ++j) {
//            delta[i * n + j] = rand() % 5 + 1;
//            delta[j * n + i] = delta[i * n + j];
//        }
//    }
//
//    std::vector<float> v_pos(pos, pos + n * 2);
//    std::vector<float> v_delta(delta, delta + n * n);
//
//    float *result_cpu = construct_lap_z_cpu(v_pos, v_delta, n);
//    float *result_gpu = construct_lap_z_gpu(v_pos, v_delta, n);
//
//    int i;
//    for (i = 0; i < n * n; ++i){
//        if (abs(result_cpu[i] - result_gpu[i]) > 0.1){
//            printf("%f %f %d %d\n", result_cpu[i], result_gpu[i], i / n, i % n);
////            break;
//        }
//    }
//
//
//    free(pos);
//    free(delta);
//    free(result_cpu);
//    free(result_gpu);
//}
