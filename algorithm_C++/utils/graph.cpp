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
    }
}

void Graph::insert_node(int node_name) {
    insert_node(std::to_string(node_name));
}
//void Graph::insert_node(int node_name) {
//
//}

bool Graph::exist_node(std::string name){
    return node_name_to_idx.find(name) != node_name_to_idx.end();
}

bool Graph::exist_node(int node_idx) {
    return node_idx < n_nodes;
}

void Graph::insert_edge(std::string src, std::string dst, float dist){
    insert_node(src);
    insert_node(dst);
    int src_idx = node_name_to_idx[src];
    int dst_idx = node_name_to_idx[dst];
    edges[src_idx][dst_idx] = dist;
    edges[dst_idx][src_idx] = dist;
}

bool Graph::exist_edge(std::string src, std::string dst) {
    int src_idx = node_name_to_idx[src];
    int dst_idx = node_name_to_idx[dst];
    return edges[src_idx].find(dst_idx) != edges[src_idx].end();
}

bool Graph::exist_edge(int src_idx, int dst_idx) {
    /*
     * src: source节点编号
     * dst: target节点编号
     */

    return edges[src_idx].find(dst_idx) != edges[src_idx].end();
}

void Graph::insert_edge(int src, int dst, float dist) {
    /*
     * src, dst: 节点名字
     */

    insert_edge(std::to_string(src), std::to_string(dst), dist);
}

float Graph::get_edge_len(std::string src_name, std::string dst_name){
    int src_idx = node_name_to_idx[src_name];
    int dst_idx = node_name_to_idx[dst_name];
    return get_edge_len(src_idx, dst_idx);
}

float Graph::get_edge_len(int src_idx, int dst_idx){
    if (!exist_edge(src_idx, dst_idx)) throw EdgeNotExistError();
    return edges[src_idx][dst_idx];
}

void Graph::weight_edge_len(){

}

void Graph::resize(int nnodes, int nedges) {
    n_nodes = nnodes;
    n_edges = nedges;
    node_cnt = 0;
    nodes.resize(nnodes);
    edges.resize(nnodes);
}

void Graph::save(const std::string path) {
//    freopen((char*)path.data(), "w", stdout);
    std::ofstream fout((char*)path.data(), std::ios::out);
    for (int i = 0; i < n_nodes; ++i){
        for (std::unordered_map<int, float>::iterator it = edges[i].begin(); it != edges[i].end(); ++it){
            if (i < it->first){
                fout << i << ' ' << it->first << '\n';
            }
        }
    }
}

int Graph::get_root(std::vector<int> &parent, int idx){
    if (parent[idx] < 0){
        return idx;
    }
    else return (parent[idx] = get_root(parent, parent[idx]));
}

void Graph::merge(std::vector<int> &parent, int root1, int root2){
    if (parent[root1] < parent[root2]){  // root1节点更多
        parent[root1] += parent[root2];
        parent[root2] = root1;
    }
    else{   // root2节点更多
        parent[root2] += parent[root1];
        parent[root1] = root2;
    }
}

bool Graph::check_connected() {
    std::vector<int> parent(n_nodes);
    std::fill(parent.begin(), parent.end(), -1);
    int nc = n_nodes;
    int src, dst;
    int rootsrc, rootdst;
    for (src = 0; src < n_nodes; ++src){
        for (std::unordered_map<int, float>::iterator it = edges[src].begin(); it != edges[src].end(); ++it){
            dst = it->first;
            rootsrc = get_root(parent, src);
            rootdst = get_root(parent, dst);
            if (rootsrc != rootdst){
                --nc;
                merge(parent, rootsrc, rootdst);
            }
        }
    }
    std::cout << "connected components: " << nc << std::endl;
    return nc == 1;
}
