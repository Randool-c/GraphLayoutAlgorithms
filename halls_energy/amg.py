# It seems that this algorithm can only be applied to connected graph
# It doesn't work well in a graph with lots of connected components

import numpy as np
import numpy.matlib
import json

from os.path import join as pjoin

from cfg import path
from . import interpolation as ip
from .get_laplacian import construct_laplacian
from data_io import read_data, write_data


def drawing(laplacian, mass_matrix, dim, interpolation_method, threshold):
    epsilon = 1e-9
    n = laplacian.shape[0]
    print(n)
    if n < threshold:
        vectors = direct_solution(laplacian, mass_matrix, dim, epsilon)
    else:
        interpolation_matrix = (ip.weighted_interpolation(laplacian)
                                if interpolation_method == 'w' else ip.edge_contraction_interpolation(laplacian))
        if interpolation_matrix.shape[0] == interpolation_matrix.shape[1]:
            vectors = direct_solution(laplacian, mass_matrix, dim, epsilon)
        else:
            vectors = (interpolation_matrix *
                       drawing(interpolation_matrix.T * laplacian * interpolation_matrix,
                               calculate_coarser_mass_matrix(interpolation_matrix, mass_matrix),
                               dim, interpolation_method, threshold))
            vectors = power_iteration(vectors, laplacian, mass_matrix, epsilon)
        print(n)
    return vectors


def calculate_coarser_mass_matrix(interpolation_matrix, mass_matrix):
    v1n = np.matlib.ones((interpolation_matrix.shape[0], 1), float)
    tmp = interpolation_matrix.T * mass_matrix * v1n
    coarser_mass_matrix = np.matlib.identity(interpolation_matrix.shape[1], float)
    for i in range(interpolation_matrix.shape[1]):
        coarser_mass_matrix[i, i] = tmp[i, 0]
    return coarser_mass_matrix


# initial_vectors: 一个矩阵，矩阵的每一列对应着某一维度的初始坐标猜测
def power_iteration(initial_vectors, laplacian, mass_matrix, epsilon):
    n = mass_matrix.shape[0]
    v1 = np.sqrt(mass_matrix) * np.matlib.ones((n, 1), float)
    v1 /= np.linalg.norm(v1)
    initial_vectors = np.insert(initial_vectors, 0, np.array(v1).flatten(), axis=1)  # 将退化解插入到第一列
    tmp_massmat = np.linalg.inv(np.sqrt(mass_matrix))
    b_mat = tmp_massmat * laplacian * tmp_massmat
    # del tmp_massmat
    b_mat = gershgorin(b_mat) * np.matlib.identity(n) - b_mat

    for i in range(1, initial_vectors.shape[1]):
        tmp = np.sqrt(mass_matrix) * initial_vectors[..., i]

        tmp /= np.linalg.norm(tmp)
        while True:
            # print('hello world')
            last_tmp = tmp
            tmp1 = tmp
            for j in range(i):
                tmp1 = tmp1 - ((tmp.T * initial_vectors[..., j])[0, 0]) * initial_vectors[..., j]
            tmp = b_mat * tmp1
            tmp = tmp / np.linalg.norm(tmp)
            if (tmp.T * last_tmp)[0, 0] > 1 - epsilon:
                break
        tmp1 = tmp
        initial_vectors[..., i] = tmp1

    initial_vectors = np.delete(initial_vectors, 0, axis=1)  # 删除退化解（第一列）
    initial_vectors = np.linalg.inv(np.sqrt(mass_matrix)) * initial_vectors
    return initial_vectors


def gershgorin(b_mat):
    ans = -float('Inf')
    n = b_mat.shape[0]
    for i in range(n):
        asum = 0
        for j in range(n):
            asum += abs(b_mat[i, j])
        asum += - abs(b_mat[i, i]) + b_mat[i, i]
        ans = max(asum, ans)
    return ans


def direct_solution(laplacian, mass_matrix, dim, epsilon):
    initial_guess = np.matlib.matrix(np.random.rand(laplacian.shape[0], dim))
    return power_iteration(initial_guess, laplacian, mass_matrix, epsilon)


class Solver:
    def __init__(self, nodes, edges, interpolate_method='w', threshold=100, target_dim=2):
        self.ip_method = interpolate_method
        self.th = threshold
        self.lap, self.mass_m = construct_laplacian(nodes, edges)
        self.n_nodes = len(nodes)
        self.dim = target_dim
        # print(self.lap)

    def execute(self):
        lap = np.matlib.matrix(self.lap)
        mass_m = np.matlib.matrix(self.mass_m)
        coordinates = drawing(lap, mass_m, self.dim, self.ip_method, self.th)
        # print(coordinates)
        return coordinates


class Lapsolver:
    def __init__(self, lap, mass):
        self.lap = lap
        self.mass = mass

    def execute(self):
        lap = np.matlib.matrix(self.lap)
        mass_m = np.matlib.matrix(self.mass)
        coordinates = drawing(lap, mass_m, 2, 'e', 50)
        # print(coordinates)
        return coordinates


def run_halls_energy():
    nodes, edges = read_data(pjoin(path.DATA_ROOT, 'dw256A', 'dw256A.mtx'))
    # nodes, edges = read_data(pjoin(path.DATA_ROOT, '1138_bus', '1138_bus.mtx'))
    print(len(nodes))
    print(len(edges))
    # nodes, edges = read_data(pjoin(path.DATA_ROOT, 'test_dataset', 'test_dataset.mtx'))
    dim = 2
    solver = Solver(nodes, edges, 'w', 50, dim)
    # solver = Lapsolver([[9, -5, 0, -4, 0], [-5, 17, -2, -7, -3], [0, -2, 4, -2, 0],
    #                            [-4, -7, -2, 19, -6], [0, -3, 0, -6, 9]], np.eye(5))
    result_x = solver.execute()
    print(result_x)
    # ans = {'nodes': result_x.tolist(), 'edges': edges}
    write_data('result.json', result_x.tolist(), edges)
