import numpy as np
import json

from utils.shortest_path import dijkstra
from sklearn.decomposition import PCA
from os.path import join as pjoin

from data_io import read_data, write_data
from cfg import path


def get_graph_list(nodes, edges):
    n_nodes = len(nodes)
    graphlist = [{} for _ in range(n_nodes)]
    for edge in edges:
        src, dst = edge
        graphlist[src][dst] = 1
        graphlist[dst][src] = 1
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


class Solver:
    def __init__(self, nodes, edges, target_dim=2, m_pivots=None):
        self.n_nodes = len(nodes)
        self.target_dim = target_dim
        if m_pivots is None:
            self.m_pivots = min(self.n_nodes, 50)
        else:
            self.m_pivots = m_pivots
        self.graphlist = get_graph_list(nodes, edges)
        for edge in edges:
            add_edge_len(self.graphlist, edge)

        print(self.graphlist)

    def choose_pivots(self):
        dist = np.full((self.n_nodes,), np.inf)
        pos = np.empty((self.n_nodes, self.m_pivots))
        p = np.random.randint(0, self.n_nodes)
        for i in range(self.m_pivots):
            p_dist = dijkstra(p, self.graphlist)
            pos[:, i] = p_dist
            dist = np.minimum(dist, p_dist)
            p = np.argmax(dist)
        return pos

    def reduce_dimensionality(self, x):
        pca_solver = PCA(n_components=self.target_dim)
        ans = pca_solver.fit_transform(x)
        return ans

    def hde(self):
        pos = self.choose_pivots()
        return self.reduce_dimensionality(pos)


def run_hde():
    nodes, edges = read_data(pjoin(path.DATA_ROOT, 'dw256A', 'dw256A.mtx'))
    # nodes, edges = read_data(pjoin(path.DATA_ROOT, '1138_bus', '1138_bus.mtx'))
    dim = 2
    solver = Solver(nodes, edges, target_dim=dim)
    result_x = solver.hde()
    write_data('result.json', result_x.tolist(), edges)
