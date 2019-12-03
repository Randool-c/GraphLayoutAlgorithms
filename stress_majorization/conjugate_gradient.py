import numpy as np
from . import cfg as settings

def cg(A, x, b):
    r = b - A @ x
    p = r
    r_at_r = r @ r
    while True:
        A_at_p = A @ p
        alpha = r_at_r / (p @ A_at_p)
        x = x + alpha * p
        newr = r - alpha * A_at_p

        # print('norm', np.linalg.norm(newr))
        if np.linalg.norm(newr) < settings.cg_iteration_terminate_epsilon:
            break

        newr_at_newr = newr @ newr
        beta = newr_at_newr / r_at_r
        p = newr + beta * p
        r = newr
        r_at_r = newr_at_newr
    # print(x)
    return x


if __name__ == '__main__':
    A = [[1, 2, 3],
         [2, -1, 2],
         [3, 2, 1]]
    x = np.random.rand(3)
    # x = np.zeros(3)
    print(x)
    A = np.array(A, dtype=float)
    b = np.array([3, 2, 9], dtype=float)
    ans = cg(A, x, b)
    print(ans)
