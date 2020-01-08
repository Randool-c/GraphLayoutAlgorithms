#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cstdlib>
//#include<helper_cuda.h>


#define BLOCK_X 32
#define BLOCK_Y 32


__global__ void compute_stress_gpu(double *pos, double *dist, double *out, int n_nodes){
    int i = threadIdx.x + blockIdx.x * BLOCK_X;
    int j = threadIdx.y + blockIdx.y * BLOCK_Y;
    int idxx = threadIdx.x;
    int idxy = threadIdx.y;

    __shared__ double shared_nums[BLOCK_X][BLOCK_Y];

    if (i == j || i >= n_nodes || j >= n_nodes){
        shared_nums[idxx][idxy] = 0;
    }
    else{
        double d = sqrt((pos[2 * i] - pos[2 * j]) * (pos[2 * i] - pos[2 * j])
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


double compute_stress(double *pos, double *dist, int n){
    int i, j;
    double ans = 0;
    for (i = 0; i < n; ++i){
        for (j = 0; j < n; ++j){
            if (i == j){
                continue;
            }
            double d = sqrt((pos[2 * i] - pos[2 * j]) * (pos[2 * i] - pos[2 * j])
                            + (pos[2 * i + 1] - pos[2 * j + 1]) * (pos[2 * i + 1] - pos[2 * j + 1]));
            d = (d / dist[i * n + j] - 1);
            d *= d;
            ans += d;
        }
    }
    return ans;
}


void f(int n){
    double *pos, *dist, out;
    double *d_pos, *d_dist, *d_out;

    pos = (double*)malloc(sizeof(double) * n * 2);
    dist = (double*)malloc(sizeof(double) * n * n);
    cudaMalloc((void**)&d_pos, sizeof(double) * n * 2);
    cudaMalloc((void**)&d_dist, sizeof(double) * n * n);
    cudaMalloc((void**)&d_out, sizeof(double));

    // initialize
    for (int i = 0; i < n; ++i){
        pos[i * 2] = rand() / double(RAND_MAX);
        pos[i * 2 + 1] = rand() / (double)(RAND_MAX);
    }
    for (int i = 0; i < n * n; ++i){
        dist[i] = double(rand() % 10) + 1;
    }

    cudaMemcpy(d_pos, pos, sizeof(double) * n * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, dist, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(double));

    int bx = 1 + (n - 1) / BLOCK_X;
    int by = 1 + (n - 1) / BLOCK_Y;
    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    compute_stress_gpu<<<dimGrid, dimBlock>>>(d_pos, d_dist, d_out, n);

    cudaMemcpy(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
    double normal_ans = compute_stress(pos, dist, n);
    printf("%f\n", out);
    printf("%f\n", normal_ans);

    cudaFree(d_out);
    cudaFree(d_pos);
    cudaFree(d_dist);
    free(pos);
    free(dist);

    cudaDeviceReset();
}

int main(){
    f(101);
}

__global__ void reduction2d(double *m, double *out, int n){
    int i = threadIdx.x + blockIdx.x * BLOCK_X;
    int j = threadIdx.y + blockIdx.y * BLOCK_Y;
    int idxx = threadIdx.x;
    int idxy = threadIdx.y;

    __shared__ double shared_nums[BLOCK_X][BLOCK_Y];
    if (i < n && j < n){
        shared_nums[idxx][idxy] = m[i * n + j];
    }
    else{
        shared_nums[idxx][idxy] = 0;
    }

    // reduction on Y
    for (unsigned int stride = BLOCK_Y / 2; stride > 0; stride /= 2){
        __syncthreads();
        if (idxy < stride){
            shared_nums[idxx][idxy] += shared_nums[idxx][idxy + stride];
        }
    }

    // reduction on X
    if (idxy == 0) {
        for (unsigned int stride = BLOCK_X / 2; stride > 0; stride /= 2) {
            __syncthreads();
            if (idxx < stride) {
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


//int main(int argc, const char **argv){
//    int n_nodes = 1024;
//    double *m, *d_m, out, *d_out;
//
//    m = (double*)malloc(sizeof(double) * n_nodes * n_nodes);
//    cudaMalloc((void**)&d_m, sizeof(double) * n_nodes * n_nodes);
//    cudaMalloc((void**)&d_out, sizeof(double));
//
//    double accu = 0;
//    for (int i = 0; i < n_nodes * n_nodes; ++i){
//        m[i] = i;
//        accu += m[i];
//    }
//
//    cudaMemcpy(d_m, m, sizeof(double) * n_nodes * n_nodes, cudaMemcpyHostToDevice);
//    cudaMemset(d_out, 0, sizeof(double));
//
//    int bx = 1 + (n_nodes - 1) / BLOCK_X;
//    int by = 1 + (n_nodes - 1) / BLOCK_Y;
//    dim3 dimGrid(bx, by);
//    dim3 dimBlock(BLOCK_X, BLOCK_Y);
//    reduction2d<<<dimGrid, dimBlock>>>(d_m, d_out, n_nodes);
//
//    cudaMemcpy(&out, d_out, sizeof(double), cudaMemcpyDeviceToHost);
//
//    printf("%f\n", out);
//    printf("%f\n", accu);
//
//    cudaFree(d_m);
//    cudaFree(d_out);
//    free(m);
//
//    cudaDeviceReset();
//    return 0;
//}
