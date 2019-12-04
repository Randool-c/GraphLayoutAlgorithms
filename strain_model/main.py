import numpy as np
import json
import matplotlib.pyplot as plt

from os.path import join as pjoin
from sklearn.manifold import MDS

from .get_dist import get_dist
from cfg import path
from data_io import read_data, write_data


class Solver:
    def __init__(self, nodes, edges, target_dim=2):
        self.dist = get_dist(nodes, edges)
        self.mds = MDS(n_components=target_dim, dissimilarity='precomputed', n_jobs=16)

    def execute(self):
        result = self.mds.fit_transform(self.dist)
        return result


def run_strain_model():
    nodes, edges = read_data(pjoin(path.DATA_ROOT, 'dw256A', 'dw256A.mtx'))
    # nodes, edges = read_data(pjoin(path.DATA_ROOT, 'test_dataset', 'test_dataset.mtx'))
    dim = 2
    solver = Solver(nodes, edges, dim)
    result_x = solver.execute()
    write_data('result.json', result_x.tolist(), edges)
    print(result_x)
