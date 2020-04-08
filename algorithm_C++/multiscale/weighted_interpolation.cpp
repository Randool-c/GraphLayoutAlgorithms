#include "weighted_interpolation.h"


namespace weighted_interpolation{
    void find_representatives(std::set<int> &representatives, mat::Mat &weights, Graph &weights){
        /*
         * @representatives: 代表元素集合
         * @weights: 图的weights矩阵，其中往往weight = 1 / (dist * dist), dist为距离矩阵
         */

        float t0 = 0.03;
        float delta_t = 0.03;
        int num_sweeps = 5;

        for (int i = 0; i < num_sweeps; ++i){

        }
    }

    mat::Mat solve_r(BaseOptimizer *optimizer, Graph &graph, int target_dim, int th){

    }

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim) {
        int n_nodes = graph.n_nodes;
        int th = 100;

//        std::vector<int> k_list;
//        generate_k_list(k_list, n_nodes, ratio, th);

//        mat::Mat pos = solve_r(optimizer, graph, target_dim, k_list, 0);
        return pos;
    }
}
