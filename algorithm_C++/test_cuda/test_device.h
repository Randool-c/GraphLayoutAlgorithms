//
// Created by chenst on 2020/4/16.
//

#ifndef MULTILEVEL_STRESS_C___TEST_DEVICE_H
#define MULTILEVEL_STRESS_C___TEST_DEVICE_H

__host__ __device__ int run_on_cpu_or_gpu();
__global__ void run_on_gpu();

#endif //MULTILEVEL_STRESS_C___TEST_DEVICE_H
