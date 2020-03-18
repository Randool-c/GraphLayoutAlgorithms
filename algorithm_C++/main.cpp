#include <iostream>

#include "layout/stress_majorization/stress.h"
#include "utils/matrix.h"
#include "utils/io.h"
#include "utils/graph.h"
#include "utils/shortest_path.hpp"
#include "multiscale/fast.h"
using namespace std;

int main() {
    Graph graph;
    io::read_dataset("1138_bus", graph);
    int target_dim = 2;
    int n_nodes = graph.n_nodes;
    BaseOptimizer *optimizer = new StressOptimizer();
    mat::Mat ans_x = fast::solve(optimizer, graph, target_dim);

//    mat::Mat dist = mat::empty(n_nodes, n_nodes);
//    shortest_path::dijkstra(dist, graph);
//    StressOptimizer optimizer(dist, target_dim);
//    mat::Mat initial_x = mat::random(graph.n_nodes, target_dim);
//    mat::Mat ans_x = optimizer.optimize(initial_x);
    ans_x.save("output.txt");
    delete optimizer;
    return 0;
}
