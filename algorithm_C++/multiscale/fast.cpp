#include "fast.h"

namespace fast {

    void kcenter(std::vector<int> &centers, std::vector<int> &nearest_center, int k, mat::Mat &shortest_dist) {
        /*
         * @centers: 一个结果vector，用来保存所有得到的centers
         * @nearest_center: 一个结果vector，用来保存距离每个节点最近的中心的编号
         */

        int init_node = 0;
        int n_nodes = shortest_dist.nr;
        centers.push_back(init_node);
        mat::Mat shortest_from_center = shortest_dist.get_row(init_node);
//    mat::Mat nearest_center = mat::fill(1, shortest_from_center.nc, init_node);
//    vector<int> nearest_center(shortest_from_center.nc);
        std::fill(nearest_center.begin(), nearest_center.end(), init_node);

        int far_node;
        for (int i = 0; i < k - 1; ++i) {
            far_node = shortest_from_center.argmax().item();
            centers.push_back(far_node);
            for (int j = 0; j < n_nodes; ++j) {
                if (shortest_dist(j, far_node) < shortest_from_center(j)) {
                    shortest_from_center(j) = shortest_dist(j, far_node);
//                nearest_center(j) = far_node;
                    nearest_center[j] = far_node;
                }
            }
        }
    }

    void generate_k_list(std::vector<int> &k_list, int n_nodes, int ratio, int th) {
        int n = th;
        do {
            k_list.push_back(n);
            n *= ratio;
        } while (n < n_nodes);
        k_list.push_back(n_nodes);
    }

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim) {
        std::srand((int)std::time(0));

        int n_nodes = graph.n_nodes;
        mat::Mat all_pair_dist = mat::empty(n_nodes, n_nodes);
        shortest_path::dijkstra(all_pair_dist, graph);
        mat::Mat initial_x = mat::random(n_nodes, target_dim);

        std::vector<int> k_list;
        generate_k_list(k_list, n_nodes);

        std::vector<int> centers;
        std::vector<int> nearest_center(n_nodes);
        mat::Mat center_dist;
        mat::Mat center_x;
        for (int k : k_list) {
            std::cout << k << std::endl;
            centers.clear();
            kcenter(centers, nearest_center, k, all_pair_dist);

            std::set<int> centers_set(centers.begin(), centers.end());
            center_dist = all_pair_dist(centers, centers);

            optimizer->initialize(center_dist, target_dim);
            center_x = initial_x.get_rows(centers);
            center_x = optimizer->optimize(center_x);
            initial_x.set_rows(centers, center_x);

            for (int i = 0; i < n_nodes; ++i){
                if (centers_set.find(i) == centers_set.end()){
                    initial_x(i, 0) = initial_x(nearest_center[i], 0) + rand() / (float)RAND_MAX;
                    initial_x(i, 1) = initial_x(nearest_center[i], 1) + rand() / (float)RAND_MAX;
                }
            }
        }
        return initial_x;
    }
}