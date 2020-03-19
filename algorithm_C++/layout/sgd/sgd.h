//
// Created by chenst on 2020/3/18.
//

#ifndef MULTILEVEL_STRESS_C___SGD_H
#define MULTILEVEL_STRESS_C___SGD_H

#include "../../utils/graph.h"
#include "../../utils/consts.h"
#include "../../utils/matrix.h"
#include "../../utils/shortest_path.hpp"
#include "../../utils/io.h"
#include "../base_optimizer.hpp"
#include <string>
#include <utility>
#include <algorithm>
#include <cmath>


class SGDOptimizer: public BaseOptimizer{
public:
    mat::Mat weights;
    mat::Mat dist;
    int n_nodes;
    int target_dim;

    SGDOptimizer();
    SGDOptimizer(mat::Mat target_dist, int target_dim=2);
    void initialize(mat::Mat target_dist, int target_dim);
    mat::Mat optimize(mat::Mat &initial_x);
    float optimize_iter(mat::Mat &pos, std::vector<std::pair<int, int>> &all_pairs, float);
    float compute_stress(mat::Mat x);
    void get_etas(std::vector<float> &etas, int t_max=30, int t_maxmax=200, float eps=0.03);
};

#endif //MULTILEVEL_STRESS_C___SGD_H
