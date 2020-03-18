#include <iostream>

#include "layout/stress_majorization/stress.h"
#include "utils/matrix.h"
#include "utils/io.h"
#include "utils/graph.h"
#include "utils/shortest_path.hpp"
#include "multiscale/fast.h"
#include <string>
#include <ctime>
using namespace std;

const string datasetname = "1138_bus";


void run_multilevel(){
    Graph graph;
    io::read_dataset(datasetname, graph);
    int target_dim = 2;
    int n_nodes = graph.n_nodes;
    BaseOptimizer *optimizer = new StressOptimizer();
    mat::Mat ans_x = fast::solve(optimizer, graph, target_dim);

//    mat::Mat dist = mat::empty(n_nodes, n_nodes);
//    shortest_path::dijkstra(dist, graph);
//    StressOptimizer optimizer(dist, target_dim);
//    mat::Mat initial_x = mat::random(graph.n_nodes, target_dim);
//    mat::Mat ans_x = optimizer.optimize(initial_x);
//    ans_x.save("output.txt");
//    graph.save("edges.txt");
//    delete optimizer;
}

void run_layout(){
    Graph graph;
    io::read_dataset(datasetname, graph);
    int target_dim = 2;
    int n_nodes = graph.n_nodes;

    mat::Mat dist = mat::empty(n_nodes, n_nodes);
    shortest_path::dijkstra(dist, graph);
    StressOptimizer optimizer(dist, target_dim);
    mat::Mat initial_x = mat::random(graph.n_nodes, target_dim);
    mat::Mat ans_x = optimizer.optimize(initial_x);
//    ans_x.save("output.txt");
//    graph.save("edges.txt");
}

int main() {
//    std::clock_t start1, end1, start2, end2;
//    start1 = std::clock();
//    run_multilevel();
//    end1 = std::clock();
//    start2 = std::clock();
//    run_layout();
//    end2 = std::clock();
//    std::cout << "multilevel time: " << end1 - start1 << std::endl;
//    std::cout << "layout time: " << end2 - start2 << std::endl;
    clock_t start, end;
    start = clock();
//    run_multilevel();
    run_layout();
    end = clock();
    cout << "cost: " << end - start << endl;
    return 0;
}
