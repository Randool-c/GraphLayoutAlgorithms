#include "weighted_interpolation.h"


namespace weighted_interpolation{
    void find_representatives(std::set<int> &representatives, mat::Mat &weights){
        /*
         * @representatives: 代表元素集合
         * @weights: 图的weights矩阵，其中往往weight = 1 / (dist * dist), dist为距离矩阵
         */

        double t = 0.05;
        double delta_t = 0.05;
        int num_sweeps = 4;
        int n_nodes = weights.nr;

        double partial_degree;
        double degree;
        double ratio;

        std::vector<int> indices(n_nodes);
        for (int i = 0; i < n_nodes; ++i) indices[i] = i;

        for (int sweep = 0; sweep < num_sweeps; ++sweep){
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
            for (int i : indices){
                if (representatives.find(i) != representatives.end()) continue;
                partial_degree = degree = 0;
                for (int j : indices){
                    if (i == j) continue;

                    if (representatives.find(j) != representatives.end()){
                        partial_degree += weights(i, j);
                    }
                    degree += weights(i, j);
                }

                ratio = partial_degree / degree;
                if (ratio < t){
                    representatives.insert(i);
                }
            }
            t += delta_t;
        }
    }

    mat::Mat construct_interpolation_matrix(mat::Mat weights, std::set<int> &representatives, std::vector<int> &representatives_v){
        /*
         * @weights: 权值矩阵
         * @n_nodes: 当前图节点数量
         * @sub_n_nodes: 粗化后的子图的节点数量，等于representatives的数量
         * @representatives: 代表元素
         */
        int n_nodes = weights.nr;
        int sub_n_nodes = representatives.size();

        mat::Mat interpolation_m = mat::zeros(n_nodes, sub_n_nodes);

        for (int i = 0; i < sub_n_nodes; ++i){
            interpolation_m(representatives_v[i], i) = 1;
        }
        double tmp_weight_sum;
        for (int i = 0; i < n_nodes; ++i){
            if (representatives.find(i) == representatives.end()){
                tmp_weight_sum = 0;
                for (int j : representatives_v){
                    tmp_weight_sum += weights(i, j);
                }
                for (int j = 0; j < sub_n_nodes; ++j){
                    interpolation_m(i, j) = weights(i, representatives_v[j]) / tmp_weight_sum;
                }
            }
        }
        return interpolation_m;
    }

    mat::Mat solve_r(BaseOptimizer *optimizer, mat::Mat dist, mat::Mat weights, int target_dim, int th){
        int n_nodes = dist.nr;
        std::cout << "n nodes: " << n_nodes << std::endl;
        if (n_nodes < th){
            mat::Mat initial_pos = mat::random(n_nodes, target_dim);
            optimizer->initialize(dist, target_dim);
            mat::Mat pos = optimizer->optimize(initial_pos);
            return pos;
        }
        else{
            std::set<int> representatives;
            find_representatives(representatives, weights);
            std::vector<int> representatives_v(representatives.begin(), representatives.end());

            mat::Mat sub_dist = dist(representatives_v, representatives_v);
            mat::Mat sub_weight = dist(representatives_v, representatives_v);
            mat::Mat pos = solve_r(optimizer, sub_dist, sub_weight, target_dim, th);
            std::cout << "refining: " << n_nodes << std::endl;

            mat::Mat interpolation_m = construct_interpolation_matrix(weights, representatives, representatives_v);
            pos = interpolation_m.mm(pos);

            optimizer->initialize(dist, target_dim);
            optimizer->optimize(pos);
            return pos;
        }
    }

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim) {
        int n_nodes = graph.n_nodes;
        int th = 100;

        mat::Mat dist = mat::empty(n_nodes, n_nodes);
        shortest_path::dijkstra(dist, graph);
        mat::Mat weights = 1 / (dist ^ 2);
        mat::Mat pos = solve_r(optimizer, dist, weights, target_dim, th);
        return pos;
    }
}
