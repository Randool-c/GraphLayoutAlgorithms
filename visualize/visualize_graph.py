import matplotlib.pyplot as plt
import json
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visualize(path='result.json'):
    with open(path, 'r') as f:
        data = json.load(f)

    pos = data['nodes']
    pos = np.array(pos)
    print(pos)
    lines = data['edges']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1])
    # for line in lines:
    #     src, dst = line
    #     x = [pos[src][0], pos[dst][0]]
    #     y = [pos[src][1], pos[dst][1]]
    #     ax.plot(x, y)
    # ax.scatter([1,2], [4,3], [5,4])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.plot([1,4], [2,3])
    plt.show()
