import numpy.matlib as matlib
import numpy as np
import random
import sys

epsilonin = 0.000001


def edge_contraction_interpolation(laplacian):
    # print(laplacian[5])
    n = laplacian.shape[0]

    # 若节点i已经匹配，则flag[i]为True
    flag = [False] * n
    pairs = []
    for i in range(n):
        if flag[i]:
            continue
        tag = False
        for j in range(n):
            if i != j and laplacian[i, j] != 0 and not flag[j]:
                flag[i] = flag[j] = True
                pairs.append([i, j])
                tag = True
                break
        if not tag:
            pairs.append([i])

    m = len(pairs)
    interpolation_matrix = np.matlib.zeros((n, m), float)
    for i in range(m):
        for node in pairs[i]:
            interpolation_matrix[node, i] = 1

    return interpolation_matrix


def weighted_interpolation(laplacian):

    # if check_laplacian_valid(laplacian) is False:
    #     print('laplacian invalid!')
    #     sys.exit()

    threshold = inc = 0.1
    n = laplacian.shape[0]
    representatives, other_nodes = find_representatives(laplacian, threshold, inc)
    m = len(representatives)
    interpolation_matrix = matlib.zeros((n, m), dtype=float)
    for i in range(m):
        interpolation_matrix[representatives[i], i] = 1
    for i in other_nodes:
        positive_degree = negative_degree = 0.
        # calculate Pi and Ni
        for j in range(m):
            if laplacian[i, representatives[j]] < 0:
                positive_degree -= laplacian[i, representatives[j]]
            else:
                negative_degree -= laplacian[i, representatives[j]]
        # calculate A[i][j]
        if positive_degree >= -negative_degree:
            for j in range(m):
                if laplacian[i, representatives[j]] < 0:
                    interpolation_matrix[i, j] = -laplacian[i, representatives[j]] / positive_degree
        else:
            for j in range(m):
                if laplacian[i, representatives[j]] > 0:
                    interpolation_matrix[i, j] = -laplacian[i, representatives[j]] / negative_degree

    return interpolation_matrix


def check_interpolation_valid(interpolation_matrix):
    n, m = interpolation_matrix.shape[0], interpolation_matrix.shape[1]
    for i in range(n):
        sumo = 0
        for j in range(m):
            sumo += interpolation_matrix[i, j]
        if abs(sumo - 1) > epsilonin:
            print(sumo)
            return False
    else:
        return True


def check_laplacian_valid(laplacian):
    n = laplacian.shape[0]
    for i in range(n):
        flag = True
        for j in range(n):
            if abs(laplacian[i, j] - 0) > epsilonin:
                flag = False
                break
        if flag:
            return False
    else:
        return True


# threshold: 初始阈值
# inc: 每次扫描后的阈值增量
def find_representatives(laplacian, threshold, inc):
    sweeps = 5
    n = laplacian.shape[0]  # 节点个数
    representatives = set()
    arr = list(range(n))
    random.shuffle(arr)
    other_nodes = set(arr)
    for i in range(sweeps):
        nodes = other_nodes.copy()
        for j in nodes:
            degree = partial_degree = 0.
            for k in range(n):
                if k != j:
                    degree += abs(laplacian[j, k])
                    if k in representatives:
                        partial_degree += abs(laplacian[j, k])
            if partial_degree / degree < threshold:
                representatives.add(j)
                other_nodes.remove(j)
        threshold += inc
    return list(representatives), list(other_nodes)


if __name__ == '__main__':
    laplacian = matlib.matrix([[9, -5, 0, -4, 0], [-5, 17, -2, -7, -3], [0, -2, 4, -2, 0],
                               [-4, -7, -2, 19, -6], [0, -3, 0, -6, 9]], dtype=float)
    # print(weighted_interpolation(laplacian))
    print(edge_contraction_interpolation(laplacian))


# if __name__ == '__main__':
#     main()

