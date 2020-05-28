//
// Created by chenst on 2020/3/22.
//

#include "nicely.h"

namespace nicely{
    void generate_k_list(std::vector<int> &k_list, int n_nodes, double ratio, int th){
        /*
         * 当新的图的节点个数小于th时
         */
        int n = n_nodes;
        do {
            n *= ratio;
            k_list.push_back(n);
        } while (n > th);
    }

    void merge_update(DisjointSet &disjoint_set, std::vector<std::set<int>> &neighbors, int src, int dst){
        int new_root = disjoint_set.merge(src, dst);
        int old_root = new_root == src ? dst : src;

        // 将old_root的所有neighbors和old_root的链接置为new_root
        for(int target : neighbors[old_root]){
            neighbors[target].erase(old_root);
            neighbors[target].insert(new_root);
//            if (target != new_root && target != old_root) {
//                std::cout << "is root: " << (disjoint_set.get_root(target) == target) << ' '
//                          << disjoint_set.parent[disjoint_set.get_root(target)] << std::endl;
//            }
        }
        neighbors[new_root].insert(neighbors[old_root].begin(), neighbors[old_root].end());
        neighbors[new_root].erase(new_root);
        neighbors[new_root].erase(old_root);
    }

    mat::Mat solve_r(BaseOptimizer *optimizer, Graph &graph, int target_dim, std::vector<int> &k_list, int k_idx){
        int n_nodes = graph.n_nodes;
        std::cout << "coarsing nodes number: " << n_nodes << std::endl;
        if (k_idx == k_list.size()){  // 达到最简图，直接进行布局
            mat::Mat initial_pos = mat::random(n_nodes, target_dim);
            mat::Mat dist = mat::empty(n_nodes, n_nodes);
            shortest_path::dijkstra(dist, graph);
            optimizer->initialize(dist, target_dim);
            mat::Mat pos = optimizer->optimize(initial_pos);
            return pos;
        }
        else {
            int k = k_list[k_idx];
            std::vector <std::set<int>> neighbors(n_nodes);
            std::vector <std::pair<int, int>> all_edges;

            DisjointSet disjoint_set(n_nodes);

            // 初始化neighbors和所有边pair
            for (int i = 0; i < graph.edges.size(); ++i) {
                for (auto &it : graph.edges[i]) {
                    if (i == it.first) continue;
                    neighbors[i].insert(it.first);
                    if (i < it.first) {
                        all_edges.push_back(std::make_pair(i, it.first));
                    }
                }
            }

            // contracting
            int rootsrc, rootdst;
            double min_cost;
            std::pair<int, int> contracted_pair;
            std::vector<int> intersection_set;
            int cluster_score, degree_score, homotopic_score, score;
            while (disjoint_set.get_n_roots() > k) {
//                std::cout << "n roots: " << disjoint_set.get_n_roots() << std::endl;
                min_cost = POS_INF;
                for (auto &pair : all_edges) {
                    rootsrc = disjoint_set.get_root(pair.first);
                    rootdst = disjoint_set.get_root(pair.second);
                    if (rootsrc == rootdst) continue;

                    cluster_score = disjoint_set.get_n_cluster_nodes(rootsrc) + disjoint_set.get_n_cluster_nodes(rootdst);
                    degree_score = std::max(neighbors[rootsrc].size(), neighbors[rootdst].size());

                    intersection_set.clear();
                    intersection_set.resize(std::max(neighbors[rootsrc].size(), neighbors[rootdst].size()));
                    std::set_intersection(neighbors[rootsrc].begin(), neighbors[rootsrc].end(),
                                          neighbors[rootdst].begin(), neighbors[rootdst].end(),
                                          intersection_set.begin());
                    homotopic_score = intersection_set.size();

                    score = cluster_score + 2 * degree_score + homotopic_score;
                    if (score < min_cost) {
                        min_cost = score;
                        contracted_pair = std::make_pair(rootsrc, rootdst);
                    }
                }
                merge_update(disjoint_set, neighbors, contracted_pair.first, contracted_pair.second);
            }

            // construct graph and solve recurssively
            mat::Mat dist = mat::empty(n_nodes, n_nodes);
            shortest_path::dijkstra(dist, graph);
            std::vector<int> roots(disjoint_set.roots.begin(), disjoint_set.roots.end());
            std::map<int, int> root_to_idx;
            std::vector<std::set<int>> clusters(k);
            Graph contracted_graph(k, 1);

            // 插入节点，保证graph中节点列表的顺序和roots中的顺序相同
            for (int i = 0; i < k; ++i){
                contracted_graph.insert_node(std::to_string(roots[i]));
                root_to_idx[roots[i]] = i;
            }

            for (int i = 0; i < n_nodes; ++i){
                clusters[root_to_idx[disjoint_set.get_root(i)]].insert(i);
            }
            double weight;
            int src_idx, dst_idx;
            std::cout << "here>? " << std::endl;
            for (int src : roots){
                for (int dst : neighbors[src]){  // 确保src和dst都为根
                    if (src < dst){
                        weight = 0;
                        for (int src_node : clusters[root_to_idx[src]]){
                            for (int dst_node : clusters[root_to_idx[dst]]){
                                weight += dist(src_node, dst_node);
                            }
                        }
                        weight /= (clusters[root_to_idx[src]].size() * clusters[root_to_idx[dst]].size());
                        contracted_graph.insert_edge(std::to_string(src), std::to_string(dst), weight);
                    }
                }
            }

            std::cout << "refining " << std::endl;
            // 得到简化后的图的布局
            mat::Mat pos = solve_r(optimizer, contracted_graph, target_dim, k_list, k_idx + 1);
            std::cout << "refining node number: " << n_nodes << std::endl;
            // 进行refine
            mat::Mat ans_pos = mat::empty(n_nodes, target_dim);
            ans_pos.set_rows(roots, pos);
            for (int i = 0; i < n_nodes; ++i){
                if (disjoint_set.roots.find(i) == disjoint_set.roots.end()){
                    ans_pos(i, 0) = ans_pos(disjoint_set.get_root(i), 0) + rand() / (double)RAND_MAX;
                    ans_pos(i, 1) = ans_pos(disjoint_set.get_root(i), 1) + rand() / (double)RAND_MAX;
                }
            }
            optimizer->initialize(dist, target_dim);
            ans_pos = optimizer->optimize(ans_pos);
            return ans_pos;
        }
    }

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim) {
        int n_nodes = graph.n_nodes;
        double ratio = 0.5;
        int th = 100;

        std::vector<int> k_list;
        generate_k_list(k_list, n_nodes, ratio, th);

        mat::Mat pos = solve_r(optimizer, graph, target_dim, k_list, 0);
        return pos;
    }
}
