import compute_lap_z as module

import numpy as np
from time import time


n = 1001


def test_correct():
    pos = np.random.rand(n, 2)
    delta = np.random.rand(n, 100)
    delta = delta @ delta.T

    ans_cpu = module.compute_lap_z_cpu(pos.ravel(), delta.ravel(), n)
    ans_gpu = module.compute_lap_z_gpu(pos.ravel(), delta.ravel(), n)

    ans_cpu = np.array(ans_cpu, copy=False).reshape(n, n)
    ans_gpu = np.array(ans_gpu, copy=False).reshape(n, n)

    print(np.max(np.abs(ans_cpu - ans_gpu)))
    print(np.max(np.abs(np.diag(ans_cpu) - np.diag(ans_gpu))))
    print(ans_cpu[0][0], ans_gpu[0][0])
    print(np.mean(np.diag(ans_cpu)))
    print(np.mean(np.diag(ans_gpu)))


def test_performance():
    repeat = 20

    pos = np.random.rand(n, 2)
    delta = np.random.rand(n, 100)
    delta = delta @ delta.T

    pos = pos.ravel()
    delta = delta.ravel()

    cpu_time = 0
    for i in range(repeat):
        pos = np.random.rand(n, 2)
        delta = np.random.rand(n, 100)
        delta = delta @ delta.T

        pos = pos.ravel()
        delta = delta.ravel()
        cpu_begin = time()
        ans_cpu = module.compute_lap_z_cpu(pos, delta, n)
        cpu_time += time() - cpu_begin
    cpu_time /= repeat

    gpu_time = 0
    for i in range(repeat):
        pos = np.random.rand(n, 2)
        delta = np.random.rand(n, 100)
        delta = delta @ delta.T

        pos = pos.ravel()
        delta = delta.ravel()
        gpu_begin = time()
        ans_cpu = module.compute_lap_z_gpu(pos, delta, n)
        gpu_time += time() - gpu_begin
    gpu_time /= repeat

    print("cpu time cost average: {:.6f}; gpu time cost average {:.6f}".format(cpu_time, gpu_time))


if __name__ == '__main__':
    test_performance()
