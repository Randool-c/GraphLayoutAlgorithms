#include "stress.h"

StressOptimizer::StressOptimizer() {

}

StressOptimizer::StressOptimizer(mat::Mat target_dist, int dim) {
    initialize(target_dist, dim);
}


void StressOptimizer::initialize(mat::Mat target_dist, int dim) {
    n_nodes = target_dist.nr;
//    dist = mat::empty(n_nodes, n_nodes);
//    weights = mat::empty(n_nodes, n_nodes);
//    construct_laplacian(graph);

//    delta = dist * weights;  // 对角线元素为nan, 不可用
//    target_dim = dim;
    dist = target_dist;
    weights = 1 / (dist ^ 2);
    lap = weights.copy();

    float row_sum;
    for (int i = 0; i < n_nodes; ++i){
        row_sum = 0;
        for (int j = 0; j < n_nodes; ++j){
            if (i == j) continue;
            row_sum += lap(i, j);
            lap(i, j) = -lap(i, j);
        }
        lap(i, i) = row_sum;
    }

    delta = dist * weights;
    target_dim = dim;
}


//void StressOptimizer::construct_laplacian(Graph &graph){
//    n_nodes = graph.n_nodes;
////    std::cout << "enter" << std::endl;
//    shortest_path::dijkstra(dist, graph);
//
//    weights = 1 / (dist ^ 2);
//
//    lap = weights.copy();
//
//    float row_sum;
//    for (int i = 0; i < n_nodes; ++i){
//        row_sum = 0;
//        for (int j = 0; j < n_nodes; ++j){
//            if (i == j) continue;
//            row_sum += lap(i, j);
//            lap(i, j) = -lap(i, j);
//        }
//        lap(i, i) = row_sum;
//    }
//}

void StressOptimizer::construct_lap_z(mat::Mat &lap_z, mat::Mat &z) {
    mat::zeros(lap_z);
    float sum_z;
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

void StressOptimizer::cg(mat::Mat &A, mat::Mat &x, mat::Mat &b){
    /*
     * solve the linear system Ax=b
     */

    float th = 1e-2;
    mat::Mat r = b - A.mm(x);
    mat::Mat p = r;
    float r_at_r = r.dot(r);

    mat::Mat A_at_p;
    float alpha;
    mat::Mat newr;
    float newr_at_newr;
    float beta;
    while (1){
        A_at_p = A.mm(p);
        alpha = r_at_r / (p.dot(A_at_p));
        x = x + alpha * p;
        newr = r - alpha * A_at_p;

        if (newr.l2_norm() < th) break;

        newr_at_newr = newr.dot(newr);
        beta = newr_at_newr / r_at_r;
        p = newr + beta * p;
        r = newr;
        r_at_r = newr_at_newr;
    }
}

float StressOptimizer::compute_stress(const mat::Mat &x) {
    float stress = 0;
    float stress_ij;
    for (int i = 0; i < n_nodes; ++i){
        for (int j = i + 1; j < n_nodes; ++j){
            stress_ij = (x[i] - x[j]).l2_norm() / dist(i, j) - 1;
            stress_ij *= stress_ij;

            stress += stress_ij;
        }
    }
    return stress;
}

mat::Mat StressOptimizer::stress_optimize_iter(mat::Mat &lap_z, mat::Mat &z){
    construct_lap_z(lap_z, z);
    mat::Mat x;
    mat::Mat b;
    mat::Mat ans_x(n_nodes, target_dim);
    for (int i = 0; i < target_dim; ++i){
        x = z.get_col(i);
        b = lap_z.mm(x);
        cg(lap, x, b);
        ans_x.set_col(i, x);
    }
    return ans_x;
}

mat::Mat StressOptimizer::optimize(mat::Mat &initial_x){
    float th = 1e-2;

    mat::Mat lap_z(n_nodes, n_nodes);
    mat::Mat z = initial_x;
    mat::Mat x = stress_optimize_iter(lap_z, z);

    float stress_z = compute_stress(z);
    float stress_x = compute_stress(x);
    std::cout << stress_z << " " << stress_x << std::endl;
    while (stress_z - stress_x >= th * stress_z){
        z = x;
        stress_z = stress_x;
        x = stress_optimize_iter(lap_z, z);
        stress_x = compute_stress(x);
        std::cout << stress_z << " " << stress_x << std::endl;
    }
    return x;
}

StressOptimizer::~StressOptimizer() { }
