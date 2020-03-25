//
// Created by chenst on 2020/3/22.
//

#ifndef MULTILEVEL_STRESS_C___NICELY_H
#define MULTILEVEL_STRESS_C___NICELY_H

#include "../utils/graph.h"
#include "../utils/matrix.h"
#include "../utils/shortest_path.hpp"
#include "../layout/base_optimizer.hpp"
#include "../utils/disjoint_set.h"
#include "../utils/consts.h"

#include<vector>
#include<algorithm>
#include<set>
#include<unordered_map>
#include<utility>
#include<cstdlib>

using namespace std;

namespace nicely{
    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim);
    void generate_k_list(std::vector<int> &k_list, int n_nodes, float ratio=0.5, int th=100);
    mat::Mat solve_r(BaseOptimizer*, Graph &, int, std::vector<int> &, int);
    void merge_update(DisjointSet&, std::vector<std::set<int>>&, int, int);

}

#endif //MULTILEVEL_STRESS_C___NICELY_H
