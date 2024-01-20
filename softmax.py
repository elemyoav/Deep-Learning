import numpy as np

def softmax(z):
    # We will stabalize the softmax function by subtracting the maximum value
    z_max = np.max(z, axis=0)
    z = z - z_max

    return np.exp(z) / np.sum(np.exp(z), axis=0)


def f(W, x, b):

    return softmax(np.dot(W.T, x) + b)

