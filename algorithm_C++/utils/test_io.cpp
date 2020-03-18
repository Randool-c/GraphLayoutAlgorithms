//
// Created by chenst on 2020/1/6.
//

#include "io.h"
#include "graph.h"
#include<iostream>
using namespace std;

int main(){
    Graph graph;
    read_dataset("dw256A", graph);
    cout << graph.n_nodes << " " << graph.n_edges << endl;
//    for (int i = 0; i < graph.n_nodes; ++i){
//        cout << graph.nodes[i] << endl;
//    }
    for (int i = 0; i < graph.n_edges; ++i){
        for (auto item : graph.edges[i]){
            cout << i << " " << item.first << " " << item.second << endl;
        }
    }
    return 0;
}
