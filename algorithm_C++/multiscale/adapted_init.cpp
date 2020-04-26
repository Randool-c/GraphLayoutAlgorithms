#include "adapted_init.h"

namespace adapted_init{

    struct dist_pair{
        int target_center;
        int dist;
        dist_pair(int t, int d): target_center(t), dist(d) {}
        dist_pair(dist_pair &&other){
            target_center = other.target_center;
            dist = other.dist;
        }
        bool operator<(const dist_pair &other) const{
            return dist < other.dist;
        }
    };

    void kcenter(std::vector<int> &centers, std::vector<std::set<dist_pair>> &dist_of_centers, int k, mat::Mat &shortest_dist) {
        /*
         * @centers: 一个结果vector，用来保存所有得到的centers
         * @dist_from_centers: 记录距离每个节点最近的一些centers，从小到大排序, 一个大小为n_nodes的数组
         */

        int init_node = 0;
        int n_nodes = shortest_dist.nr;
        centers.push_back(init_node);
        mat::Mat shortest_from_center = shortest_dist.get_row(init_node);

        int far_node;
        for (int i = 0; i < k - 1; ++i) {
            far_node = shortest_from_center.argmax().item();
            centers.push_back(far_node);
            for (int j = 0; j < n_nodes; ++j) {
                dist_of_centers[j].insert(dist_pair(far_node, shortest_dist(far_node, j)));
            }
        }
    }

    float get_lr(int step_cnt){
        static float k = (1 / max_step) * std::log(lr_min);
        return lr_max * std::exp(k * step_cnt);
    }

    mat::Mat solve_r(BaseOptimizer *optimizer, mat::Mat dist, mat::Mat weights, int target_dim, std::vector<int> &k_list, int k_idx){
        int n_nodes = dist.nr;
        std::cout << "n nodes: " << n_nodes << std::endl;
        if (k_idx == k_list.size()){
            mat::Mat initial_pos = mat::random(n_nodes, target_dim);
            optimizer->initialize(dist, target_dim, &weights);
            mat::Mat pos = optimizer->optimize(initial_pos);
            return pos;
        }
        else{
            int k = k_list[k_idx];
            std::vector<std::set<dist_pair>>> dist_of_centers(n_nodes);
            std::vector<int> centers;
            kcenter(centers, dist_of_centers, k, dist);
            std::set<int> centers_set(centers.begin(), centers.end());

            mat::Mat sub_dist = dist(centers, centers);
            mat::Mat sub_weights = weights(centers, centers);
            mat::Mat sub_pos = solve_r(optimizer, sub_dist, sub_weights, target_dim, k_list, k_idx + 1);

            mat::Mat ans_pos = mat::random(n_nodes, target_dim);
            ans_pos.set_rows(centers, pos);

            // 确定非representatives的位置，通过stress energy求导迭代得到
            float lr;
            float mu, dx, dy, mag, r, rx, ry;
            int src, dst;
            for (int i = 0; i < n_nodes; ++i){
                if (centers_set.find(i) != centers_set.end()) continue;

                std::vector<int> required_centers;
                for (dist_pair it : dist_of_centers[i]){
                    if (it.dist <= 10) required_centers.push_back(it.target_center);
                    else break;
                }

                for (int step_cnt = 0; step_cnt < max_step; ++step_cnt) {
                    for (int target : required_centers) {
                        lr = get_lr(step_cnt);
                        mu = lr * weights(i, target);

                        dx = ans_pos(i, 0) - ans_pos(target, 0);
                        dy = ans_pos(i, 1) - ans_pos(target, 1);
                        mag = std::sqrt(dx * dx + dy * dy);
                        delta = mu * (mag - dist(i, target)) / 2;
                        r = delta / mag;

                        rx = r * dx;
                        ry = r * dy;
                        ans_pos(i, 0) = ans_pos(i, 0) - rx;
                        ans_pos(i, 1) = ans_pos(i, 1) - ry;
                    }
                    std::random_shuffle(required_centers.begin(), required_centers.end());
                }
            }

            optimizer->initialize(dist, target_dim, &weights);
            ans_pos = optimizer->optimize(ans_pos);
            return ans_pos;
        }
    }

    void generate_k_list(std::vector<int> &k_list, int n_nodes, int ratio, int th) {
        int n = n_nodes;
        do {
            n *= ratio;
            k_list.push_back(n);
//            n *= ratio;
        } while (n > th);
    }

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim){
        int n_nodes = graph.n_nodes;
        int th = 100;
        float ratio = 0.5;

        mat::Mat dist = mat::empty(n_nodes, n_nodes);
        shortest_path::dijkstra(dist, graph);

        mat::Mat weights = 1 / (dist ^ 2);

        std::vector<int> k_list;
        generate_k_list(k_list, n_nodes, ratio, th);

        mat::Mat pos = solve_r(optimizer, dist, weights, target_dim, k_list, 0);
        return pos;
    }
}
