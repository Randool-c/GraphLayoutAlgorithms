import numpy as np
import os
from os.path import join as pjoin


def make_data(name, n_nodes, edge_mean, edge_std, write_path):
    nodes = np.arange(n_nodes) + 1
    edges = []
    for src in nodes:
        n_edges = np.maximum(1, np.random.normal(edge_mean, edge_std))
        n_edges = int(n_edges)
        for _ in range(n_edges):
            dst = np.random.randint(1, n_nodes + 1)
            edges.append((src, dst, np.random.rand()))
    print(check_connected(nodes, edges))
    with open(write_path, 'w') as f:
        f.write('%%{}\n'.format(name))
        f.write('{} {} {}\n'.format(n_nodes, n_nodes, len(edges)))
        for edge in edges:
            f.write('{} {} {}\n'.format(edge[0], edge[1], edge[2]))
        f.close()


class DisjointSet:
    def __init__(self, nodes, edges):
        self.n = len(nodes)
        self.n_components = self.n
        self.node_to_idx = {v: k for k, v in enumerate(nodes)}
        self.parent = [-1] * (self.n + 1)
        self.edges = edges

        for src, dst, _ in edges:
            srcidx = self.node_to_idx[src]
            dstidx = self.node_to_idx[dst]
            self.merge(srcidx, dstidx)

    def get_root(self, idx):
        if self.parent[idx] < 0:
            return idx
        else:
            self.parent[idx] = self.get_root(self.parent[idx])
            return self.parent[idx]

    def merge(self, node1, node2):
        root1 = self.get_root(node1)
        root2 = self.get_root(node2)
        if root1 == root2:
            return

        self.n_components -= 1
        if self.parent[root1] < self.parent[root2]:
            self.parent[root1] += self.parent[root2]
            self.parent[root2] = root1
        else:
            self.parent[root2] += self.parent[root1]
            self.parent[root1] = root2


def check_connected(nodes, edges):
    disjoint_set = DisjointSet(nodes, edges)
    return disjoint_set.n_components == 1


if __name__ == '__main__':
    name = 'custom_dataset'
    if not os.path.isdir(pjoin('dataset', name)):
        os.makedirs(pjoin('dataset', name))
    make_data(name, 3000, 7, 5, pjoin('dataset', name, '{}.mtx'.format(name)))
