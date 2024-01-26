import numpy as np

def col_mean(M):
    return np.mean(M, axis=1, keepdims=True)