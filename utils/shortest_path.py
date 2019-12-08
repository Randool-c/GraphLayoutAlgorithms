import numpy as np

from time import time
from queue import PriorityQueue


def floyed(adj_matrix):
    n = adj_matrix.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if adj_matrix[i][k] + adj_matrix[k][j] < adj_matrix[i][j]:
                    adj_matrix[i][j] = adj_matrix[i][k] + adj_matrix[k][j]
    return adj_matrix


def dijkstra(source, graphlist):
    """compute the distances between the source node and the rest nodes.
        :param source:
        :param graphlist: a adjacent table for the graph, formatted as [{dst1: -1 or length, dst2: -1 or length}}, ]
        :return: a list
    """

    n_nodes = len(graphlist)
    dist = [np.inf] * n_nodes
    flag = [False] * n_nodes
    n_left = n_nodes

    heap = PriorityQueue()
    heap.put((0, source))
    while n_left > 0:
        # print(dist)
        nearest_dist, nearest_node = heap.get()
        if flag[nearest_node]:
            continue

        flag[nearest_node] = True
        n_left -= 1
        dist[nearest_node] = nearest_dist

        for dst_node, value in graphlist[nearest_node].items():
            if dist[nearest_node] + value < dist[dst_node]:
                dist[dst_node] = dist[nearest_node] + value
                heap.put((dist[dst_node], dst_node))
    return dist


def dijkstra_all(graphlist):
    n_nodes = len(graphlist)
    dist = []
    for i in range(n_nodes):
        dist.append(dijkstra(i, graphlist))
    return np.array(dist, dtype=float)


if __name__ == '__main__':
    ############### test floyed #################
    import numpy as np
    m = [[0, 2, 4, np.inf, np.inf],
         [2, 0, 3, 1, 5],
         [4, 3, 0, np.inf, 1],
         [np.inf, 1, np.inf, 0, 4],
         [np.inf, 5, 1, 4, 0]]
    m = np.array(m)

    ############## test dijkstra #################
    graphlist = [
        {1: 2, 2: 4},
        {0: 2, 2: 3, 3: 1, 4: 5},
        {0: 4, 1: 3, 4: 1},
        {1: 1, 4: 4},
        {1: 5, 2: 1, 3: 4}
    ]
    print(dijkstra_all(graphlist))
    print(floyed(m))
    print(np.all(dijkstra_all(graphlist)==floyed(m)))
