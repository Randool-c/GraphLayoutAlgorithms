//
// Created by chenst on 2019/12/9.
//

#ifndef CUDA_COMPUTE_STRESS_H
#define CUDA_COMPUTE_STRESS_H

#include<vector>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<cstdlib>


float compute_stress_cpu(std::vector<float>&, std::vector<float>&, int);
float compute_stress_gpu(std::vector<float>&, std::vector<float>&, int);

#endif //CUDA_COMPUTE_STRESS_H
