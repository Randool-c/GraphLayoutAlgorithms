import matplotlib.pyplot as plt
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize3d(x, y, z, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    for edge in edges:
        src, dst = edge
        xpos = [x[src], x[dst]]
        ypos = [y[src], y[dst]]
        zpos = [z[src], z[dst]]
        ax.plot(xpos, ypos, zpos)
    plt.show()


def visualize2d(x, y, edges):
    for edge in edges:
        src, dst = edge
        xpos = [x[src], x[dst]]
        ypos = [y[src], y[dst]]
        plt.plot(xpos, ypos, c='gray', zorder=10)
    plt.scatter(x, y, c='blue', zorder=20)
    plt.show()


def visualize(path='result.json'):
    with open(path, 'r') as f:
        data = json.load(f)

    pos = data['nodes']
    pos = np.array(pos)
    edges = data['edges']
    dim = pos.shape[1]
    if dim == 2:
        visualize2d(pos[:, 0], pos[:, 1], edges)
    elif dim == 3:
        visualize3d(pos[:, 0], pos[:, 1], pos[:, 2], edges)


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([1, 2], [4, 3], [5, 4])
    ax.plot([1, 2], [4, 3], [5, 4])
    plt.show()
