//
// Created by chenst on 2020/1/1.
//

#ifndef MULTILEVEL_STRESS_C___GRAPH_H
#define MULTILEVEL_STRESS_C___GRAPH_H

#include<vector>
#include<map>
#include<unordered_map>
#include<string>

class Graph{
public:
    int n_nodes;
    int n_edges;
    std::vector<std::string> nodes;
    std::vector<std::unordered_map<int, int>> edges;
    std::map<std::string, int> node_name_to_idx;

    Graph(): n_nodes(0), n_edges(0){ }
    Graph(Graph &other){ }
    void insert_node(std::string node_name){}
    bool exist_node(std::string node_name){}
    void insert_edge(std::string src, std::string dst){}
};

#endif //MULTILEVEL_STRESS_C___GRAPH_H
