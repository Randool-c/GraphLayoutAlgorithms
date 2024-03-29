//
// Created by chenst on 2020/3/18.
//

#ifndef MULTILEVEL_STRESS_C___BASECLASS_H
#define MULTILEVEL_STRESS_C___BASECLASS_H

#include "../utils/matrix.h"

class BaseOptimizer{
public:
    virtual ~BaseOptimizer() {}
    virtual mat::Mat optimize(mat::Mat &initial_x) {}
    virtual void initialize(mat::Mat, int, mat::Mat *w=NULL) {}
    virtual float compute_stress(mat::Mat x) {}
//    virtual float compute_stress(const mat::Mat&) {}
};

#endif //MULTILEVEL_STRESS_C___BASECLASS_H
