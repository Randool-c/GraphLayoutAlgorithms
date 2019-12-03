import numpy as np

from .shortest_path import floyed


def get_graph_list(nodes, edges):
    n_nodes = len(nodes)
    graphlist = [{} for _ in range(n_nodes)]
    for edge in edges:
        src, dst = edge
        graphlist[src][dst] = -1
        graphlist[dst][src] = -1
    return graphlist


def add_edge_len(graph_list, target_edge):
    """append 'weight' attribute for each input edge
        :param graph_list: a adjacent table for the graph, formatted as [{dst1: -1 or length, dst2: -1 or length}}, ]
        :param target_edge: (src, dst)
    """

    src, dst = target_edge
    src_adj = set(graph_list[src].keys())
    dst_adj = set(graph_list[dst].keys())
    length = len(src_adj.union(dst_adj)) - len(src_adj.intersection(dst_adj))
    graph_list[src][dst] = length
    graph_list[dst][src] = length
    return length


def get_adj_matrix(graphlist):
    n_nodes = len(graphlist)
    adj_matrix = np.full((n_nodes, n_nodes), np.inf)
    adj_matrix[np.arange(n_nodes), np.arange(n_nodes)] = 0
    for src in range(n_nodes):
        for dst in graphlist[src]:
            adj_matrix[src][dst] = graphlist[src][dst]
            adj_matrix[dst][src] = graphlist[dst][src]
    return adj_matrix


def construct_laplacian(nodes, edges):
    n_nodes = len(nodes)
    graphlist = get_graph_list(nodes, edges)
    # print(graphlist)
    for edge in edges:
        add_edge_len(graphlist, edge)
    print(graphlist)
    adj_len_matrix = get_adj_matrix(graphlist)
    dist = floyed(adj_len_matrix)  # dist[i][j] represents the expected distance between i and j
    weights = 1 / np.square(dist)

    lap = -weights.copy()
    lap[np.arange(n_nodes), np.arange(n_nodes)] = 0
    lap[np.arange(n_nodes), np.arange(n_nodes)] = -np.sum(lap, axis=1)
    return lap, dist, weights
