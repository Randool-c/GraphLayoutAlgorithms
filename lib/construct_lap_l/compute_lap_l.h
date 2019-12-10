//
// Created by chenst on 2019/12/9.
//

#ifndef CUDA_COMPUTE_LAP_L_H
#define CUDA_COMPUTE_LAP_L_H

#include<vector>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<memory.h>

float *construct_lap_z_cpu(std::vector<float>&, std::vector<float>&, int);
float *construct_lap_z_gpu(std::vector<float>&, std::vector<float>&, int);

#endif //CUDA_COMPUTE_LAP_L_H
