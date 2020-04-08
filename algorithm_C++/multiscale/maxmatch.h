//
// Created by chenst on 2020/4/5.
//

#ifndef MULTILEVEL_STRESS_C___MAXMATCH_H
#define MULTILEVEL_STRESS_C___MAXMATCH_H

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

namespace maxmatch{

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim = 2);
    void find_embedding(std::vector<int> &embedded_idx, std::map<int, std::pair<int, int>> &contracted, Graph &graph);
    mat::Mat solve_r(BaseOptimizer *optimizer, Graph &graph, int target_dim, int th);

    }

#endif //MULTILEVEL_STRESS_C___MAXMATCH_H
