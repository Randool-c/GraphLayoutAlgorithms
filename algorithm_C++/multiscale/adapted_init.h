#ifndef ALGORITHMS_ADAPTED_INIT_H
#define ALGORITHMS_ADAPTED_INIT_H

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
#include<cmath>
#include<ctime>

namespace adapted_init{
    const double lr_max = 10;
    const double lr_min = 10;
    const double max_step = 40;
    const int max_effective_center_dist = 20;  // stress迭代更新时考虑的最远的center距离
    const int th = 100;
    const double ratio = 0.33;

    struct dist_pair{
        int target_center;
        int dist;
        dist_pair(int t, int d): target_center(t), dist(d) {}
        dist_pair(const dist_pair &other){
            target_center = other.target_center;
            dist = other.dist;
        }
        bool operator<(const dist_pair &other) const{
            return dist < other.dist;
        }
    };

    void kcenter(std::vector<int> &centers, std::vector<std::set<dist_pair>> &dist_of_centers, int k, mat::Mat &shortest_dist);
    double get_lr(int step_cnt);
    mat::Mat solve_r(BaseOptimizer *optimizer, mat::Mat dist, mat::Mat weights, int target_dim, std::vector<int> &k_list, int k_idx);
    void generate_k_list(std::vector<int> &k_list, int n_nodes, int ratio, int th) ;
    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim);
}

#endif //ALGORITHMS_ADAPTED_INIT_H
