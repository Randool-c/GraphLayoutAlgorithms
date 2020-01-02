//
// Created by chenst on 2020/1/1.
//

#include "graph.h"

Graph::Graph(Graph &other){
    n_nodes = other.n_nodes;
    n_edges = other.n_edges;
    nodes = other.nodes;
    edges = other.edges;
}

void Graph::insert_node(std::string node_name){
    if (node_name_to_idx.find(node_name) == node_name_to_idx.end()){
        node_name_to_idx[node_name] = n_nodes;
        n_nodes += 1;
        nodes.push_back(node_name);
    }
}

bool Graph::exist_node(std::string name){
    return node_name_to_idx.find(name) != node_name_to_idx.end();
}

void Graph::insert_edge(std::string src, std::string dst){
    insert_node(src);
    insert_edge(dst);
    int src_idx = node_name_to_idx[src];
    int dst_idx = node_name_to_idx[dst];
    edges[src_idx][dst_idx] = 1;
    edges[dst_idx][src_idx] = 1;
}



