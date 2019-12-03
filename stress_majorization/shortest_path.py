def floyed(adj_matrix):
    n = adj_matrix.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if adj_matrix[i][k] + adj_matrix[k][j] < adj_matrix[i][j]:
                    adj_matrix[i][j] = adj_matrix[i][k] + adj_matrix[k][j]
    return adj_matrix


if __name__ == '__main__':
    ############### test floyed #################
    import numpy as np
    m = [[0, 2, 4, np.inf, np.inf],
         [2, 0, 3, 1, 5],
         [4, 3, 0, np.inf, 1],
         [np.inf, 1, np.inf, 0, 4],
         [np.inf, 5, 1, 4, 0]]
    m = np.array(m)
    print(np.all(m.T == m))
    print(m)
    # print(np.all(m.T == m))
    print(m.T)
    ans = floyed(m)
    print('ans\n')
    print(ans)
