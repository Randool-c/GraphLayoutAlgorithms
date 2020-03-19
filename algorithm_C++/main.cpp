#include <iostream>

#include "layout/stress_majorization/stress.h"
#include "utils/matrix.h"
#include "utils/io.h"
#include "utils/graph.h"
#include "utils/shortest_path.hpp"
#include "multiscale/fast.h"
#include "layout/sgd/sgd.h"
#include <string>
#include <ctime>
using namespace std;

const string datasetname = "lshp2233";


void run_multilevel(){
    Graph graph;
    io::read_dataset(datasetname, graph);
    if (!graph.check_connected()) throw NotFullyConnectedError();

    int target_dim = 2;
    int n_nodes = graph.n_nodes;
    BaseOptimizer *optimizer = new SGDOptimizer();
    mat::Mat ans_x = fast::solve(optimizer, graph, target_dim);

    ans_x.save("output.txt");
    graph.save("edges.txt");
    delete optimizer;
}

void run_layout(){
    Graph graph;
    io::read_dataset(datasetname, graph);

    if (!graph.check_connected()) throw NotFullyConnectedError();

    int target_dim = 2;
    int n_nodes = graph.n_nodes;

    mat::Mat dist = mat::empty(n_nodes, n_nodes);
    shortest_path::dijkstra(dist, graph);
    SGDOptimizer optimizer(dist, target_dim);
    mat::Mat initial_x = mat::random(graph.n_nodes, target_dim);
//    initial_x.save("initial_x.txt");
    mat::Mat ans_x = optimizer.optimize(initial_x);
    ans_x.save("output.txt");
    graph.save("edges.txt");
}

int main() {
    clock_t start, end;
    start = clock();
    run_multilevel();
//    run_layout();
    end = clock();
    cout << "cost: " << end - start << endl;
    return 0;
}
