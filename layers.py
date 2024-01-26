import numpy as np
from matplotlib import pyplot as plt
from utils import col_mean
class ReLULayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
    
    def relu(self, Z):
        return np.maximum(0, Z)

    def forward(self, X):
        Z = np.dot(self.W, X) + self.b
        return self.relu(Z)

    def JacxMv(self, x, v):
        z = np.dot(self.W, x) + self.b
        dReLU_dZ = col_mean(z > 0)
        J = np.diag(dReLU_dZ.flatten()) @ self.W

        return J @ v
    
    def JacTMx(self, X, V):
        # Jacobian of ReLU with respect to X times V
        Z = np.dot(self.W, X) + self.b
        dReLU_dZ = col_mean(Z > 0)
        return np.dot(self.W.T, dReLU_dZ * V)

    def JacTMW(self, X, V):
        # Jacobian of ReLU with respect to W times V
        Z = np.dot(self.W, X) + self.b
        dReLU_dZ = Z > 0
        return np.dot(dReLU_dZ * V, X.T)

    def JacTMb(self, X, V):
        # Jacobian of ReLU with respect to b times V
        Z = np.dot(self.W, X) + self.b
        dReLU_dZ = col_mean(Z > 0)
        return dReLU_dZ * V
    
    def JacobianTest(self, x):
        d = np.random.randn(*x.shape)
        d = d / np.linalg.norm(d)
        eps = 0.1 
        E = []
        E2 = []

        for _ in range(10):
            y1 = np.linalg.norm(self.forward(x + eps * d) - self.forward(x))
            y2 = np.linalg.norm(self.forward(x + eps * d) - self.forward(x) - self.JacxMv(x, eps * d))
            E.append(y1)
            E2.append(y2)
            eps = eps * 0.5
        
        plt.plot(E, label='first order error')
        plt.plot(E2, label='second order error')
        plt.yscale('log')
        plt.legend()
        plt.show()
        
if __name__ == '__main__':
    l = ReLULayer(3, 6)
    X = np.array([[1,2,3], [4,5,6], [7,8,9]]).T
    C = np.array([[0, 0, 1, 0]]).T
    l.JacobianTest(X)