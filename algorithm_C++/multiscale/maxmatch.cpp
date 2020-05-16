#include "maxmatch.h"

namespace maxmatch{
    void find_embedding(std::vector<int> &embedded_idx, std::map<int, std::pair<int, int>> &contracted, Graph &graph){
        /*
         * 找到当前图中某节点contracted后属于粗化子图中的哪个节点，假设原图有n个节点，粗化后的图有m个节点
         * @embedded_idx: 大小为n的数组，embedded_idx[i]代表节点i在contracted后属于粗化后的图中的节点编号
         * @contracted: 一个逆映射代表着每一个粗化后的节点包含有原来图中的哪些节点
         */

        int n_nodes = graph.n_nodes;
        int subgraph_node_cnt = 0;
        std::vector<bool> selected(n_nodes, false);  // 节点是否已经contracted
        for (int i = 0; i < n_nodes; ++i){
            if (selected[i]) continue;
            bool matched = false;
            for (auto &it : graph.edges[i]){
                if (selected[it.first] || i >= it.first) continue;
                embedded_idx[i] = embedded_idx[it.first] = subgraph_node_cnt;
                contracted[subgraph_node_cnt] = std::make_pair(i, it.first);
                matched = true;
                ++subgraph_node_cnt;
                selected[i] = selected[it.first] = true;
                break;
            }

            if (!matched){
                embedded_idx[i] = subgraph_node_cnt;
                contracted[subgraph_node_cnt] = std::make_pair(i, i);
                ++subgraph_node_cnt;
                selected[i] = true;
            }
        }
    }

    mat::Mat solve_r(BaseOptimizer *optimizer, Graph &graph, int target_dim, int th){
        int n_nodes = graph.n_nodes;
        std::cout << "n nodes: " << n_nodes << std::endl;
        if (n_nodes < th){
               mat::Mat initial_pos = mat::random(n_nodes, target_dim);
               mat::Mat dist = mat::empty(n_nodes, n_nodes);
               shortest_path::dijkstra(dist, graph);
               optimizer->initialize(dist, target_dim);
               mat::Mat pos = optimizer->optimize(initial_pos);
               return pos;
        }
        else {
            std::vector<int> embedded_idx(n_nodes);
            std::map <int, std::pair<int, int>> contracted;
            find_embedding(embedded_idx, contracted, graph);

            mat::Mat dist = mat::empty(n_nodes, n_nodes);
            shortest_path::dijkstra(dist, graph);

            int subgraph_n_nodes = contracted.size();
            Graph subgraph(subgraph_n_nodes, 0);
            // 插入节点
            for (int i = 0; i < subgraph_n_nodes; ++i){
                subgraph.insert_node(i);
            }

            // 插入边
            int src, dst;
            double tmp_edge_weights;
            for (int i = 0; i < n_nodes; ++i){
                for (auto &it : graph.edges[i]){
                    if (i >= it.first) continue;
                    src = embedded_idx[i];
                    dst = embedded_idx[it.first];
                    if (!subgraph.exist_edge(src, dst)){
                        tmp_edge_weights = 0;

                        tmp_edge_weights += dist(contracted[src].first, contracted[dst].first);
                        tmp_edge_weights += dist(contracted[src].first, contracted[dst].second);
                        tmp_edge_weights += dist(contracted[src].second, contracted[dst].first);
                        tmp_edge_weights += dist(contracted[src].second, contracted[dst].second);

                        subgraph.insert_edge(src, dst, tmp_edge_weights / 4);
                    }
                }
            }

            mat::Mat pos = solve_r(optimizer, subgraph, target_dim, th);
            // refine
            std::cout << "n nodes: " << n_nodes << std::endl;
            mat::Mat ans_pos = mat::empty(n_nodes, target_dim);
            double dx, dy;
            for (int i = 0; i < subgraph_n_nodes; ++i){
                dx = (rand() / (double)RAND_MAX - 0.5) * 2;
                dy = (rand() / (double)RAND_MAX - 0.5) * 2;
                if (contracted[i].first == contracted[i].second){
                    ans_pos(contracted[i].first, 0) = pos(i, 0) + dx;
                    ans_pos(contracted[i].first, 1) = pos(i, 1) + dy;
                }
                else{
                    ans_pos(contracted[i].first, 0) = pos(i, 0) + dx;
                    ans_pos(contracted[i].first, 1) = pos(i, 1) + dy;
                    ans_pos(contracted[i].second, 0) = pos(i, 0) - dx;
                    ans_pos(contracted[i].second, 1) = pos(i, 1) - dy;
                }
            }
            optimizer->initialize(dist, target_dim);
            ans_pos = optimizer->optimize(ans_pos);
            return ans_pos;
        }
    }

    mat::Mat solve(BaseOptimizer *optimizer, Graph &graph, int target_dim){
        int th = 100;
        int n_nodes = graph.n_nodes;

        mat::Mat pos = solve_r(optimizer, graph, target_dim, th);
        return pos;
    }
}
