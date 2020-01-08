//
// Created by chenst on 2020/1/7.
//

#ifndef MULTILEVEL_STRESS_C___STRESS_H
#define MULTILEVEL_STRESS_C___STRESS_H

#include "../utils/graph.h"
#include "../utils/consts.h"
#include "../utils/matrix.hpp"
#include "../utils/shortest_path.hpp"
#include "../utils/io.h"
#include <string>


void construct_laplacian(Graph &);
void stress_optimize(mat::Mat<float> &);

class StressOptimizer{
public:
    Mat<float> lap;
    Mat<float> dist;
    Mat<float> weights;
    Mat<float> delta;
    Graph graph;
    int n_nodes;
    int target_dim;

    StressOptimizer(std::string);
    void construct_laplacian(Graph &);
    void construct_lap_z(mat::Mat<float> &dst_lap_z, mat::Mat<float> &z);
    void operator()(mat::Mat &);
};
#endif //MULTILEVEL_STRESS_C___STRESS_H
