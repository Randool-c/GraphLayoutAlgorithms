import matplotlib.pyplot as plt
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.plot(x, y, z)
    plt.show()


def visualize2d(x, y):
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.show()


def visualize(path='result.json'):
    with open(path, 'r') as f:
        data = json.load(f)

    pos = data['nodes']
    pos = np.array(pos)
    dim = pos.shape[1]
    if dim == 2:
        visualize2d(pos[:, 0], pos[:, 1])
    elif dim == 3:
        visualize3d(pos[:, 0], pos[:, 1], pos[:, 2])


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([1, 2], [4, 3], [5, 4])
    ax.plot([1, 2], [4, 3], [5, 4])
    plt.show()
