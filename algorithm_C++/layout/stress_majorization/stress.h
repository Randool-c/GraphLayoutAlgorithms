//
// Created by chenst on 2020/1/7.
//

#ifndef MULTILEVEL_STRESS_C___STRESS_H
#define MULTILEVEL_STRESS_C___STRESS_H

#include "../../utils/graph.h"
#include "../../utils/consts.h"
#include "../../utils/matrix.h"
#include "../../utils/shortest_path.hpp"
#include "../../utils/io.h"
#include "../base_optimizer.hpp"
#include<iostream>
#include <string>


class StressOptimizer: public BaseOptimizer{
public:
    mat::Mat lap;
    mat::Mat dist;
    mat::Mat weights;
    mat::Mat delta;
//    Graph graph;
    int n_nodes;
    int target_dim;

    StressOptimizer();
    StressOptimizer(mat::Mat target_dist, int, mat::Mat *w=NULL);
    void initialize(mat::Mat, int, mat::Mat *w=NULL);
//    void construct_laplacian(Graph &);
    void construct_lap_z(mat::Mat &dst_lap_z, mat::Mat &z);
    void operator()(mat::Mat &);
    void cg(mat::Mat &, mat::Mat &, mat::Mat &);
    float compute_stress(mat::Mat);
    mat::Mat stress_optimize_iter(mat::Mat &, mat::Mat &);
    mat::Mat optimize(mat::Mat &initial_x);
    ~StressOptimizer();
};
#endif //MULTILEVEL_STRESS_C___STRESS_H
