import numpy as np
import json
import matplotlib.pyplot as plt

from os.path import join as pjoin

from .get_laplacian import construct_laplacian
from . import cfg as settings
from cfg import path
from .conjugate_gradient import cg


class Solver:
    def __init__(self, nodes, edges, target_dim=3):
        self.lap, self.dist, self.weights = construct_laplacian(nodes, edges)
        print(np.all(self.lap==self.lap.T))
        print('laplacian\n', self.lap)
        print('distance\n', self.dist)
        print('weights\n', self.weights)
        # print(self.lap)
        # print(np.all(self.lap.T == self.lap))
        # print(np.mean((self.lap.T - self.lap).abs()))
        # print(np.max(np.abs(self.lap.T - self.lap)))
        # print(np.mean(np.abs(self.lap.T - self.lap)))
        # print('maxvalue;m', np.max(self.lap))
        # print('meanvalue', np.mean(self.lap))

        np.savetxt('lap.txt', self.lap)
        self.delta = self.dist * self.weights
        self.delta = np.where(np.isnan(self.delta), 0, self.delta)
        self.n_nodes = len(nodes)
        self.target_dim = target_dim

    def compute_stress(self, x):
        stress = 0
        # print(x)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    continue

                stress_ij = (np.linalg.norm(x[i] - x[j]) / self.dist[i, j] - 1) ** 2
                # print(stress_ij)
                stress += stress_ij
        return stress

    def terminate(self, x, z):
        stress_x = self.compute_stress(x)
        stress_z = self.compute_stress(z)
        print(stress_x, stress_z)
        return stress_z - stress_x < settings.stress_optimize_terminate_epsilon * stress_z

    def stress_optimize(self, initial_x):
        z = initial_x
        x = self.stress_optimize_iter(z)
        while not self.terminate(x, z):
            z = x
            x = self.stress_optimize_iter(z)
        return x

    def compute_l_z(self, z):
        lap_z = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            sum_z = 0
            for j in range(self.n_nodes):
                if i == j:
                    continue

                lap_z[i, j] = -self.delta[i, j] * (1 / np.linalg.norm(z[i] - z[j]))
                if np.isnan(lap_z[i, j]):
                    lap_z[i, j] = 1
                sum_z += lap_z[i, j]
            lap_z[i, i] = -sum_z
        return lap_z

    def stress_optimize_iter(self, z):
        lap_z = self.compute_l_z(z)
        ans_x = np.empty_like(z)
        for i in range(self.target_dim):
            x = z[:, i]
            b = lap_z @ z[:, i]
            ans_x[:, i] = cg(self.lap, x, b)
            # r = b - self.lap @ x
            # p = r.copy()
            # r_at_r = r @ r
            # while True:
            #     lap_at_p = self.lap @ p
            #     alpha = r_at_r / (p @ lap_at_p)
            #     x = x + alpha * p
            #     new_r = r - alpha * lap_at_p
            #
            #     norm = np.linalg.norm(new_r)
            #     print(norm)
            #     if np.isnan(norm):
            #         raise Exception()
            #     if norm < settings.cg_iteration_terminate_epsilon:
            #         break
            #
            #     new_r_at_new_r = new_r @ new_r
            #     beta = new_r_at_new_r / r_at_r
            #     p = new_r + beta * p
            #
            #     r = new_r
            #     r_at_r = new_r_at_new_r
            ans_x[:, i] = x
        return ans_x


def read_data(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    data = [x.split() for x in data[1:]]
    n_nodes = int(data[0][0])
    n_edges = int(data[0][2])

    nodes = list(range(1, 1 + n_nodes))
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    edges = []
    for edge in data[1:]:
        src = int(edge[0])
        dst = int(edge[1])
        edges.append([node_to_idx[src], node_to_idx[dst]])
    return nodes, edges


def run_stress_model():
    nodes, edges = read_data(pjoin(path.DATA_ROOT, 'dw256A', 'dw256A.mtx'))
    # nodes, edges = read_data(pjoin(path.DATA_ROOT, 'test_dataset', 'test_dataset.mtx'))
    n_nodes = len(nodes)
    solver = Solver(nodes, edges, 3)
    initial_x = np.random.rand(n_nodes, 2)
    result_x = solver.stress_optimize(initial_x)
    print(result_x)
    ans = {'nodes': result_x.tolist(), 'edges': edges}
    with open('result.json', 'w') as f:
        json.dump(ans, f, indent=4)
        f.close()
