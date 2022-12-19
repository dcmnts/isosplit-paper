import numpy as np


def evaluate_clustering(labels1: np.array, labels2: np.array):
    K1 = int(np.max(labels1))
    K2 = int(np.max(labels2))
    if (K1 == 0) or (K2 == 0):
        return 0
    CM = np.zeros((K1, K2))
    for i in range(len(labels1)):
        if (labels1[i] > 0) and (labels2[i] > 0):
            CM[int(labels1[i]) - 1, int(labels2[i]) - 1] += 1
    A = np.zeros((K1, K2))
    for k1 in range(K1):
        for k2 in range(K2):
            denom = np.sum(CM[k1, :]) + np.sum(CM[:, k2]) - CM[k1, k2]
            if denom > 0:
                A[k1, k2] = CM[k1, k2] / denom
    return np.average([np.max(A[k1, :]) for k1 in range(K1)])