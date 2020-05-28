//
// Created by chenst on 2020/3/17.
//

#ifndef MULTILEVEL_STRESS_C___FAST_H
#define MULTILEVEL_STRESS_C___FAST_H

#include "../utils/graph.h"
#include "../utils/matrix.h"
#include "../utils/shortest_path.hpp"
#include "../layout/base_optimizer.hpp"
#include<vector>
#include<algorithm>
#include<set>
#include<cstdlib>
#include<ctime>
#include<stack>

namespace fast {

    const double ratio = 0.5;

    const int th = 100;

    void kcenter(std::vector<int> &, std::vector<int> &, int k, mat::Mat &);

    void generate_k_list(std::vector<int> &, int n_nodes);

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim = 2);

}

#endif //MULTILEVEL_STRESS_C___FAST_H
