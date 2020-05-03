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
#include<string>
#include<utility>
#include<exception>
#include"matrix.h"

class EdgeNotExistError: public std::exception{
public:
    const char *what() const noexcept override {
        return "Error! Target edge not exist\n";
    }
};

class Graph{
public:
    int n_nodes;
    int n_edges;
    int node_cnt;
    std::vector<std::string> nodes;
    std::vector<std::unordered_map<int, double>> edges; // first: node no, second: edge length
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
    void insert_node(int node_name);
    bool exist_node(std::string node_name);
    bool exist_node(int node_name);
    void insert_edge(std::string src, std::string dst, double dist=1.0);
    void insert_edge(int src, int dst, double dist);
    bool exist_edge(std::string src, std::string dst);
    bool exist_edge(int src, int dst);
    double get_edge_len(int, int);
    double get_edge_len(std::string, std::string);
    void weight_edge_len();
    void save(const std::string path);
//    Mat get_laplacian(){}
//    Mat get_shortest_path(){}
    bool check_connected();
    int get_root(std::vector<int> &parent, int);
    void merge(std::vector<int>&, int, int);
};

#endif //MULTILEVEL_STRESS_C___GRAPH_H
