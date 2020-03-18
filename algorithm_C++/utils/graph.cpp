//
// Created by chenst on 2020/1/1.
//

#include "graph.h"

Graph::Graph(Graph &other){
    n_nodes = other.n_nodes;
    n_edges = other.n_edges;
    node_cnt = other.node_cnt;
    nodes = other.nodes;
    edges = other.edges;
}

void Graph::insert_node(std::string node_name){
    if (!exist_node(node_name)){
        node_name_to_idx[node_name] = node_cnt;
        nodes[node_cnt] = node_name;
        node_cnt += 1;
//        nodes.push_back(node_name);
    }
}

bool Graph::exist_node(std::string name){
    return node_name_to_idx.find(name) != node_name_to_idx.end();
}

void Graph::insert_edge(std::string src, std::string dst, float dist){
    insert_node(src);
    insert_node(dst);
    int src_idx = node_name_to_idx[src];
    int dst_idx = node_name_to_idx[dst];
    edges[src_idx][dst_idx] = dist;
    edges[dst_idx][src_idx] = dist;
}

void Graph::weight_edge_len(){

}

void Graph::resize(int nnodes, int nedges) {
    n_nodes = nnodes;
    n_edges = nedges;
    node_cnt = 0;
    nodes.resize(nnodes);
    edges.resize(nedges);
}

void Graph::save(const std::string path) {
    freopen((char*)path.data(), "w", stdout);
    for (int i = 0; i < n_nodes; ++i){
        for (std::unordered_map<int, float>::iterator it = edges[i].begin(); it != edges[i].end(); ++it){
            if (i < it->first){
                std::cout << i << ' ' << it->first << '\n';
            }
        }
    }
    fclose(stdout);
}

//void Graph::print() {
//    std::cout << "hello world" << std::endl;
//}

//Mat Graph::get_laplacian(){
//    mat::Mat<float> lap;
//    mat::zeros(lap);
//    int sum_row;
//    int idx;dc ..
//    exit
//    for (int i = 0; i < n_nodes; ++i){
//        sum_row = 0;
//        for (int j = 0; j < n_nodes; ++j){
//            if (i == j) continue;
//            idx = i * n_nodes + j;
//            if (edges[i].find(j) != edges[i].end()){
//                sum_row += edges[i][j];
//                lap(i, j) = -edges[i][j]
//                lap(j, i) = -edges[i][j];
//            }
//        }
//        lap(i, i) = sum_row;
//    }
//    return lap;
//}
