#ifndef MULTILEVEL_STRESS_C___SHORTEST_PATH
#define MULTILEVEL_STRESS_C___SHORTEST_PATH

#include "graph.h"
#include "matrix.h"
#include "consts.h"
#include<vector>
#include<queue>
#include<algorithm>
#include<iostream>

namespace shortest_path{

struct dist_pair{
    int node;
    double dist;
    dist_pair(int x, double y): node(x), dist(y) {}
    bool operator>(const dist_pair &other) const{
        return dist > other.dist;
    }
};

inline mat::Mat dijkstra_single(int source, Graph &graph){
    int n_nodes = graph.n_nodes;
    int n_left = n_nodes;
//    std::vector dist(n_nodes, `consts`::POS_INF);

    mat::Mat row(1, n_nodes);
    double *dist = row.arr->array;
    for (int i = 0; i < n_nodes; ++i){
        dist[i] = POS_INF;
    }
    std::vector<bool> flag(n_nodes, false);

    std::priority_queue<dist_pair, std::vector<dist_pair>, std::greater<dist_pair>> heap;
    heap.push(dist_pair(source, 0));
    int nearest_node;
    int nearest_dist;
    while (n_left > 0){
        nearest_node = heap.top().node;
        nearest_dist = heap.top().dist;
        heap.pop();
        if (flag[nearest_node]) continue;

        flag[nearest_node] = true;
        --n_left;
        dist[nearest_node] = nearest_dist;
        for (auto &item : graph.edges[nearest_node]){
            if (dist[nearest_node] + item.second < dist[item.first]){
                dist[item.first] = dist[nearest_node] + item.second;
                heap.push(dist_pair(item.first, dist[item.first]));
            }
        }
    }
    return row;
}

inline void dijkstra(mat::Mat &dist, Graph &graph){
    int n_nodes = graph.n_nodes;
    int n_left;
//    mat::Mat dist(n_nodes, n_nodes);
    for (int i = 0; i < n_nodes * n_nodes; ++i){
        dist.arr->array[i] = POS_INF;
    }

    int nearest_node;
    int nearest_dist;
//    std::cout << graph.n_nodes << " nodes " << graph.n_edges << " edges " << std::endl;
    for (int source = 0; source < n_nodes; ++source){
        std::priority_queue<dist_pair, std::vector<dist_pair>, std::greater<dist_pair>> heap;
        std::vector<bool> flag(n_nodes, false);
        n_left = n_nodes;

        heap.push(dist_pair(source, 0));
        while (n_left > 0){
            nearest_node = heap.top().node;
            nearest_dist = heap.top().dist;
            heap.pop();

            if (flag[nearest_node]) continue;

            flag[nearest_node] = true;
            --n_left;
            dist(source, nearest_node) = nearest_dist;
            for (auto &item : graph.edges[nearest_node]){
                if (dist(source, nearest_node) + item.second < dist(source, item.first)){
                    dist(source, item.first) = dist(source, nearest_node) + item.second;
                    heap.push(dist_pair(item.first, dist(source, item.first)));
                }
            }
        }
    }
}

inline mat::Mat floyed(){

}

}

#endif