/*
 * 使用找到图的某一个极大匹配作为粗化的子图
 */

#include <iostream>

#include "layout/stress_majorization/stress.h"
#include "utils/matrix.h"
#include "utils/io.h"
#include "utils/graph.h"
#include "utils/shortest_path.hpp"
#include "multiscale/fast.h"
#include "multiscale/nicely.h"
#include "layout/sgd/sgd.h"
#include "multiscale/maxmatch.h"
#include "multiscale/weighted_interpolation.h"
#include "multiscale/adapted_init.h"
#include <string>
#include <ctime>
using namespace std;

//const string datasetname = "custom_dataset";
//const string datasetname = "bcspwr10";
const string datasetname = "dw256A";
// const string datasetname = "crystk02";
//const string datasetname = "lshp2233";
//const string datasetname = "1138_bus";


void run_multilevel(){
    Graph graph;
    io::read_dataset(datasetname, graph);
    if (!graph.check_connected()) throw NotFullyConnectedError();

    int target_dim = 2;
    int n_nodes = graph.n_nodes;
    BaseOptimizer *optimizer = new SGDOptimizer();
    mat::Mat ans_x = adapted_init::solve(optimizer, graph, target_dim);

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

//int main(){
//    run_layout();
//}

//int main(){
//    clock_t start_multiscale, end_multiscale;
//    clock_t start_layout, end_layout;
//    int t_multiscale, t_layout;
//
//    start_multiscale = clock();
//    run_multilevel();
//    end_multiscale = clock();
//    t_multiscale = (double)(end_multiscale - start_multiscale) / CLOCKS_PER_SEC;
//
//    start_layout = clock();
//    run_layout();
//    end_layout = clock();
//    t_layout = (double)(end_layout - start_layout) / CLOCKS_PER_SEC;
//
//    std::cout << "加上Multiscale算法后耗时：" << t_multiscale << "秒" << std::endl;
//    std::cout << "直接进行布局耗时：" << t_layout << "秒" << std::endl;
//}

int main() {
    int test_times = 10;
    double accu_seconds = 0;

    for (int i = 0; i < test_times; ++i) {
        clock_t start, end;
        start = clock();
        run_multilevel();
//        run_layout();
        end = clock();
        cout << "cost: " << (double) (end - start) / CLOCKS_PER_SEC << "秒" << endl;
        accu_seconds += (double) (end - start) / CLOCKS_PER_SEC;
    }
    cout << "average cost: " << accu_seconds / test_times << endl;
    return 0;
}
