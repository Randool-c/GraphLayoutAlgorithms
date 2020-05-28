//
// Created by chenst on 2020/1/6.
//

#include "graph.h"
#include "matrix.h"
#include "shortest_path.hpp"
#include "consts.h"
#include "io.h"
#include<string>
using namespace std;


const string datasetname = "1138_bus";


int main(){
//    cout << "program begins" << endl;
//    Graph graph(5, 10);
    std::string a = "0";
    Graph graph;
    io::read_dataset(datasetname, graph);
    graph.save("graph.txt");
//    graph.insert_node(a);
//    graph.insert_node("1");
//    graph.insert_node("2");
//    graph.insert_node("3");
//    graph.insert_node("4");
//
//    graph.insert_edge(a, "1", 2);
//    graph.insert_edge(a, "2", 4);
//    graph.insert_edge(a, "2", 2);
//    graph.insert_edge("1", "2", 3);
//    graph.insert_edge("1", "3", 1);
//    graph.insert_edge("1", "4", 5);
//    graph.insert_edge("2", "0", 4);
//    graph.insert_edge("2", "1", 3);
//    graph.insert_edge("2", "4", 1);
//    graph.insert_edge("3", "4", 4);



    int n = graph.n_nodes;
    mat::Mat dist(n, n);
    shortest_path::dijkstra(dist, graph);
    dist.save("dist.txt");
    return 0;
}
