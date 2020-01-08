#include "stress.h"


StressOptimizer::StressOptimizer(std::string datasetname) {
    io::read_dataset(datasetname, graph);
    construct_laplacian();
    n_nodes = graph.n_nodes;
    delta = dist * weights;  // 对角线元素为nan, 不可引用
    target_dim = 2;
}


void StressOptimizer::construct_laplacian(Graph &graph){
    int n_nodes = graph.n_nodes;
    shortest_path::dijkstra(dist, graph);
    dist.print();

    mat::Mat<float> TMP;
    TMP = dist.square();
    weights = 1 / TMP;

    lap = weights.copy();

    float row_sum;
    for (int i = 0; i < n_nodes; ++i){
        row_sum = 0;
        for (int j = 0; j < n_nodes; ++j){
            row_sum += lap(i, j);
            lap(i, j) = -lap(i, j);
        }
        lap(i, i) = row_sum;
    }
}

void StressOptimizer::construct_lap_z(mat::Mat<float> &lap_z, mat::Mat<float> &z) {
    mat::zeros(lap_z);
    float sum_z;
    mat::Mat<float> TMP;
    for (int i = 0; i < n_nodes; ++i){
        sum_z = 0;
        for (int j = 0; j < n_nodes; ++j){
            if (i == j) continue;

            lap_z(i, j) = -delta(i, j) * (1 / (z[i] - z[j]).l2_norm());
            sum_z -= lap_z(i, j);
        }
        lap_z(i, i) = sum_z;
    }
}

void StressOptimizer::cg(mat::Mat<float> &A, mat::Mat<float> &x, mat::Mat<float> &b){
    /*
     * solve the linear system Ax=b
     */

    mat::Mat<float> TMP;
    mat::Mat<float> r;
    mat::Mat<float> p;
    mat::Mat<float> newr;
    TMP = A.mm(x);
    r = b - TMP;
    p = r;
}

void StressOptimizer::stress_optimize_iter(mat::Mat<float> &lap_z, mat::Mat<float> &z, mat::Mat<float> &ans_x){
    construct_lap_z(lap_z, z);
    mat::Mat<float> x;
    mat::Mat<float> b;
    for (int i = 0; i < target_dim; ++i){
        x = z.get_col(i);
        b = lap_z.mm(x);

    }
}

void stress_optimize(mat::Mat<float> &initial_x){
    mat::Mat<float> z = initial_x;

}
