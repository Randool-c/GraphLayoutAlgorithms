//
// Created by chenst on 2020/3/18.
//

#include "sgd.h"


SGDOptimizer::SGDOptimizer() {}

SGDOptimizer::SGDOptimizer(mat::Mat target_dist, int target_dim, mat::Mat *w) {
    initialize(target_dist, target_dim, w);
}

void SGDOptimizer::initialize(mat::Mat target_dist, int dim, mat::Mat *w) {
    n_nodes = target_dist.nr;
    dist = target_dist;
    if (w){
        weights = *w;
    }
    else{
        weights = 1 / (dist ^ 2);
    }
    target_dim = dim;
}

double SGDOptimizer::compute_stress(mat::Mat x){
    double stress = 0;
    double stress_ij;
    for (int i = 0; i < n_nodes; ++i){
        for (int j = i + 1; j < n_nodes; ++j){
            stress_ij = (x[i] - x[j]).l2_norm() / dist(i, j) - 1;
            stress_ij *= stress_ij;

            stress += stress_ij;
        }
    }
    return stress;
}

void SGDOptimizer::get_etas(std::vector<double> &etas, int t_max, int t_maxmax, double eps) {
    double min_weight = POS_INF;
    double max_weight = NEG_INF;
    for (int i = 0; i < n_nodes; ++i){
        for (int j = i + 1; j < n_nodes; ++j){
            min_weight = std::min(min_weight, weights(i, j));
            max_weight = std::max(max_weight, weights(i, j));
        }
    }
    double eta_max = 1 / min_weight;
    double eta_min = eps / max_weight;
    double l = std::log(eta_max / eta_min) / (t_max - 1);

    double eta_switch = 1 / max_weight;
    double eta;
    for (int t = 0; t < t_maxmax; ++t){
        eta = eta_max * std::exp(-l * t);
        if (eta < eta_switch) break;

        etas.push_back(eta);
    }
    int tau = etas.size();
    for (int t = tau; t < t_maxmax; ++t){
        eta = eta_switch / (1 + l * (t - tau));
        etas.push_back(eta);
    }
}

double SGDOptimizer::optimize_iter(mat::Mat &pos, std::vector<std::pair<int, int>> &all_pairs, double eta) {
    /*
     * 一轮更新过程，会直接修改传入的pos对象
     */
    std::random_shuffle(all_pairs.begin(), all_pairs.end());
    int src, dst;
    double w_ij, d_ij, mu, dx, dy, mag, delta, r, r_x, r_y;
    double max_delta = 0;
    for (std::pair<int, int> &p : all_pairs){
        src = p.first;
        dst = p.second;

        w_ij = weights(src, dst);
        d_ij = dist(src, dst);

        mu = eta * w_ij;
        mu = std::min(mu, double(1));

        dx = pos(src, 0) - pos(dst, 0);
        dy = pos(src, 1) - pos(dst, 1);
        mag = std::sqrt(dx * dx + dy * dy);
        delta = mu * (mag - d_ij) / 2;
        max_delta = std::max(max_delta, delta);

        r = delta / mag;
        r_x = r * dx;
        r_y = r * dy;
        pos(src, 0) = pos(src, 0) - r_x;
        pos(src, 1) = pos(src, 1) - r_y;
        pos(dst, 0) = pos(dst, 0) + r_x;
        pos(dst, 1) = pos(dst, 1) + r_y;
    }
//    std::cout << max_delta << std::endl;
//    std::cout << "stress: " << compute_stress(pos) << std::endl;
    return max_delta;
}

mat::Mat SGDOptimizer::optimize(mat::Mat &initial_x) {
    double stop_th = 0.03;  // 当所有节点的最大移动距离小于这个值时算法停止
    // 生成所有节点对
    std::vector<std::pair<int, int>> all_pairs;
    for (int src = 0; src < n_nodes; ++src){
        for (int dst = src + 1; dst < n_nodes; ++dst){
            all_pairs.push_back(std::make_pair(src, dst));
        }
    }

    // 生成所有的eta
    std::vector<double> etas;
    get_etas(etas);

    double max_delta;
    for (double eta : etas){
        max_delta = optimize_iter(initial_x, all_pairs, eta);
        std::cout << "stress: " << compute_stress(initial_x) << " max delta: " << max_delta << std::endl;
        if (max_delta < stop_th) break;
    }
    return initial_x;
}
