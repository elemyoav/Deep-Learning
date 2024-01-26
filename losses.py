import numpy as np
from utils import col_mean

class SoftmaxLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)

    def forward(self, X):
        Z = np.dot(X.T, self.W)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def grad_x(self, X, C):
        return col_mean(np.dot(self.W, (self.forward(X) - C.T).T))
      
    def grad_w(self, X, C):
        return np.dot(X, self.forward(X) - C.T)
    
    def loss(self, X, C):
        return np.sum(-np.log(self.forward(X)) * C.T) / X.shape[1]

    def grad_test_x(self, x, c):
        import matplotlib.pyplot as plt
        E = []
        E2 = []
        eps = 1
        d = np.random.randn(x.shape[0], 1)
        d = d / np.linalg.norm(d)

        for _ in range(15):
            y1 = np.linalg.norm(self.loss(x + eps * d, c) - self.loss(x, c))
            y2 = np.linalg.norm(self.loss(x + eps * d, c) - (self.loss(x, c) + eps * d.T @ self.grad_x(x, c)))
            E.append(y1)
            E2.append(y2)
            eps = eps * 0.5
        
        plt.plot(E, label='first order error')
        plt.plot(E2, label='second order error')
        plt.yscale('log')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4,5,6]]).T
    C = np.array([[0, 0, 1, 0], [1,0,0,0]]).T
    l = SoftmaxLayer(3, 4)
    l.grad_test_x(X, C)

    # losses = []
    # for _ in range(100):
    #     dW = l.grad_w(X, C)
    #     l.W -= 0.1 * dW
    #     losses.append(l.loss(X, C))
    # import matplotlib.pyplot as plt
    # plt.plot(losses)
    # plt.show()