# It seems that this algorithm can only be applied to connected graph
# It doesn't work well in a graph with lots of connected components

import numpy as np
import numpy.matlib
import json

from . import interpolation as ip


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
    def __init__(self):
        pass


# interpolate_method: 'w' -- weighted interpolation, 'e' -- edge contraction interpolation
# *d: 参数数量取决于layout的维度。
# 每一个维度是一个两个元素的list，为该维度的起始值和终止值。例如[100, 1000]
def ace(interpolate_method, *d, threshold=100):
    n = len(_input)
    # with open('dataset/data.json') as file:
    #     _input = json.load(file)["items"]

    get_index = {}  # 建立一个由IP索引到index的字典
    for i in range(n):
        get_index[_input[i]['id']] = i

    data = transform.transf(_input)
    print("transform completed")
    laplacian = np.matlib.zeros((n, n), float)
    mass_matrix = np.matlib.identity(n, float)
    for edge in data:
        if edge['sourceIP'] in get_index and edge['destinationIP'] in get_index:
            i, j = get_index[edge['sourceIP']], get_index[edge['destinationIP']]
            mass_matrix[i, i] += 1
            mass_matrix[j, j] += 1
            laplacian[i, j] -= 1
            laplacian[j, i] -= 1

    for i in range(n):
        for j in range(n):
            if i != j:
                laplacian[i, i] -= laplacian[i, j]

    coordinates = drawing(laplacian, mass_matrix, dim, interpolate_method, threshold)

    # 当节点很多时，节点的各个维度的坐标趋近于0
    # 使节点显得比较分散开
    # for i in range(dim):
    #     max_value = np.amax(coordinates[..., i])
    #     min_value = np.amin(coordinates[..., i])
    #     distance = max_value - min_value
    #     coordinates[..., i] = 1.9 / distance * coordinates[..., i] - 0.95 * (max_value + min_value) / distance

    for i in range(n):
        for j in range(dim):
            _input[i]['dim' + str(j + 1)] = 0.5 * (d[j][1] - d[j][0]) * coordinates[i, j] + 0.5 * (d[j][0] + d[j][1])

    ax1.scatter(coordinates[..., 0].getA().flatten(), coordinates[..., 1].getA().flatten(), c='r', marker='o')
    plt.show()
    return _input


# def check_laplacian(laplacian):
#     n = laplacian.shape[0]
#     for i in range(n):
#         flag = True
#         for j in range(n):
#             if laplacian[i, j] != 0:
#                 flag = False
#                 break
#         if flag:
#             return False
#     return True


# if __name__ == '__main__':
def run_halls_energy():
    ans = ace('w', [0, 1], [0, 1])

    output_json = {'items': ans}
    with open('nodes_output.json', 'w') as file:
        json.dump(output_json, file, indent=4)
