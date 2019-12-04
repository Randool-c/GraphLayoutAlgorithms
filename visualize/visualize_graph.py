import matplotlib.pyplot as plt
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize3d(x, y, z, edges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    line_x = []
    line_y = []
    line_z = []
    for edge in edges:
        src, dst = edge
        line_x.append(x[src])
        line_y.append(y[src])
        line_z.append(z[src])
    ax.plot(line_x, line_y, line_z)
    plt.show()


def visualize2d(x, y, edges):
    plt.scatter(x, y)
    # plt.plot(x, y)
    line_x = []
    line_y = []
    for edge in edges:
        src, dst = edge
        line_x.append(x[src])
        line_y.append(y[src])
    plt.plot(line_x, line_y)
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
