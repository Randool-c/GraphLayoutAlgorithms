//
// Created by chenst on 2020/1/1.
//

#ifndef MULTILEVEL_STRESS_C___GRAPH_H
#define MULTILEVEL_STRESS_C___GRAPH_H

#include<vector>
#include<map>
#include<unordered_map>
#include<string>
#include<iostream>
#include<fstream>
#include"matrix.h"

class Graph{
public:
    int n_nodes;
    int n_edges;
    int node_cnt;
    std::vector<std::string> nodes;
    std::vector<std::unordered_map<int, float>> edges; // first: node no, second: edge length
    std::map<std::string, int> node_name_to_idx;

    Graph(): n_nodes(0), n_edges(0), node_cnt(0){}
    Graph(int nnodes, int nedges): n_nodes(nnodes), n_edges(nedges){
        nodes.resize(nnodes);
        edges.resize(nnodes);
        node_cnt = 0;
    }
    Graph(std::vector<std::string> &nodes);
    Graph(Graph &);
    void resize(int nnodes, int nedges);
    void clear();
    void insert_node(std::string node_name);
    bool exist_node(std::string node_name);
    void insert_edge(std::string src, std::string dst, float dist=1.0);
    void weight_edge_len();
    void save(const std::string path);
//    Mat get_laplacian(){}
//    Mat get_shortest_path(){}
    bool check_connected();
    int get_root(std::vector<int> &parent, int);
    void merge(std::vector<int>&, int, int);
};

#endif //MULTILEVEL_STRESS_C___GRAPH_H
