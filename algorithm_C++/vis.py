import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin

ROOT = "/home/chenst/projects/Vis/Main/algorithms"
datasetname = '1138_bus'


def paint():
    pos = np.loadtxt('output.txt')
    edges = np.loadtxt('edges.txt')
    plt.scatter(pos[:, 0], pos[:, 1])
    for edge in edges:
        srcidx, dstidx = edge
        srcidx = int(srcidx)
        dstidx = int(dstidx)
        plt.plot([pos[srcidx][0], pos[dstidx][0]], [pos[srcidx][1], pos[dstidx][1]])
    plt.show()


if __name__ == '__main__':
    paint()
