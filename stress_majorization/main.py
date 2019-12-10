import numpy as np
import json
import matplotlib.pyplot as plt

from os.path import join as pjoin

from .get_laplacian import construct_laplacian
from . import cfg as settings
from cfg import path
from .conjugate_gradient import cg
from data_io import read_data
from lib import compute_stress, compute_lap_z

from time import time


class Solver:
    def __init__(self, nodes, edges, target_dim=3):
        # self.graphlist = get_graph_list(nodes, edges)
        self.lap, self.dist, self.weights = construct_laplacian(nodes, edges)

        # np.savetxt('lap.txt', self.lap)
        self.delta = self.dist * self.weights
        self.delta = np.where(np.isnan(self.delta), 0, self.delta)
        self.n_nodes = len(nodes)
        self.target_dim = target_dim

    def compute_stress(self, x):
        # begin = time()
        # stress = 0
        # print(x)
        # for i in range(self.n_nodes):
        #     for j in range(self.n_nodes):
        #         if i == j:
        #             continue
        #
        #         stress_ij = (np.linalg.norm(x[i] - x[j]) / self.dist[i, j] - 1) ** 2
        #         # print(stress_ij)
        #
        #         stress += stress_ij
        stress = compute_stress.compute_stress_gpu(x.ravel(), self.dist.ravel(), self.n_nodes)
        # print("cost {}".format(time() - begin))
        return stress

    def stress_optimize_iter(self, z):
        lap_z = self.compute_l_z(z)
        ans_x = np.empty_like(z)
        for i in range(self.target_dim):
            x = z[:, i]
            b = lap_z @ z[:, i]
            ans_x[:, i] = cg(self.lap, x, b)
        return ans_x

    def compute_l_z(self, z):
        # lap_z = np.zeros((self.n_nodes, self.n_nodes))
        # for i in range(self.n_nodes):
        #     sum_z = 0
        #     for j in range(self.n_nodes):
        #         if i == j:
        #             continue
        #
        #         lap_z[i, j] = -self.delta[i, j] * (1 / np.linalg.norm(z[i] - z[j]))
        #         if np.isnan(lap_z[i, j]):
        #             lap_z[i, j] = 1
        #         sum_z += lap_z[i, j]
        #     lap_z[i, i] = -sum_z
        lap_z = compute_lap_z.compute_lap_z_gpu(z.ravel(), self.delta.ravel(), self.n_nodes)
        lap_z = np.array(lap_z, copy=False).reshape(self.n_nodes, self.n_nodes)
        return lap_z

    def stress_optimize(self, initial_x):
        z = initial_x
        x = self.stress_optimize_iter(z)
        stress_z = self.compute_stress(z)
        stress_x = self.compute_stress(x)
        while not (stress_z - stress_x < settings.stress_optimize_terminate_epsilon * stress_z):
            z = x
            stress_z = stress_x
            x = self.stress_optimize_iter(z)
            stress_x = self.compute_stress(x)
            print(stress_z, stress_x)
        return x


def run_stress_model():
    nodes, edges = read_data(pjoin(path.DATA_ROOT, 'dw256A', 'dw256A.mtx'))
    # nodes, edges = read_data(pjoin(path.DATA_ROOT, '1138_bus', '1138_bus.mtx'))
    # nodes, edges = read_data(pjoin(path.DATA_ROOT, 'test_dataset', 'test_dataset.mtx'))
    n_nodes = len(nodes)
    dim = 2
    solver = Solver(nodes, edges, dim)
    initial_x = np.random.rand(n_nodes, dim)
    # initial_x = np.zeros((n_nodes, dim))
    result_x = solver.stress_optimize(initial_x)
    ans = {'nodes': result_x.tolist(), 'edges': edges}
    with open('result.json', 'w') as f:
        json.dump(ans, f, indent=4)
        f.close()
