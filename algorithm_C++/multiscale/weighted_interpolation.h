//
// Created by chenst on 2020/4/7.
//

#ifndef MULTILEVEL_STRESS_C___WEIGHTED_INTERPOLATION_H
#define MULTILEVEL_STRESS_C___WEIGHTED_INTERPOLATION_H

#include "../utils/graph.h"
#include "../utils/matrix.h"
#include "../utils/shortest_path.hpp"
#include "../layout/base_optimizer.hpp"
#include<vector>
#include<algorithm>
#include<set>
#include<cstdlib>
#include<ctime>
#include<utility>
#include<iostream>
#include<random>
#include<chrono>

namespace weighted_interpolation{
    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim);
    mat::Mat solve_r(BaseOptimizer *optimizer, mat::Mat dist, mat::Mat weights, int target_dim, int th);
    mat::Mat construct_interpolation_matrix(mat::Mat weights, std::set<int> &representatives, std::vector<int> &representatives_v);
    void find_representatives(std::set<int> &representatives, mat::Mat &weights);
}

#endif //MULTILEVEL_STRESS_C___WEIGHTED_INTERPOLATION_H
