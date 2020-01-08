import numpy as np


def get_graph_list(nodes, edges):
    n_nodes = len(nodes)
    graphlist = [{} for _ in range(n_nodes)]
    for edge in edges:
        src, dst = edge
        graphlist[src][dst] = -1
        graphlist[dst][src] = -1
    return graphlist


def add_edge_len(graph_list, target_edge, uniform=False):
    """append 'weight' attribute for each input edge
        :param graph_list: a adjacent table for the graph, formatted as [{dst1: -1 or length, dst2: -1 or length}}, ]
        :param target_edge: (src, dst)
        :param uniform: whether to use uniform edge length
    """

    src, dst = target_edge
    src_adj = set(graph_list[src].keys())
    dst_adj = set(graph_list[dst].keys())
    if uniform:
        length = 1
    else:
        length = len(src_adj.union(dst_adj)) - len(src_adj.intersection(dst_adj))
    graph_list[src][dst] = length
    graph_list[dst][src] = length
    return length


def construct_laplacian(nodes, edges):
    n_nodes = len(nodes)
    graphlist = get_graph_list(nodes, edges)
    # print(graphlist)
    for edge in edges:
        add_edge_len(graphlist, edge, uniform=True)

    lap = np.zeros((n_nodes, n_nodes))
    for src in range(n_nodes):
        for dst in graphlist[src]:
            lap[src][dst] = -1 / (graphlist[src][dst] * graphlist[src][dst])
            lap[dst][src] = lap[src][dst]
    lap[np.arange(n_nodes), np.arange(n_nodes)] = -np.sum(lap, axis=1)
    mass_matrix = np.zeros_like(lap)
    mass_matrix[np.arange(n_nodes), np.arange(n_nodes)] = np.diag(lap)
    print(mass_matrix)
    return lap, mass_matrix
