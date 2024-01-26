import numpy as np
from matplotlib import pyplot as plt

class ReLULayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
    
    def relu(self, Z):
        return np.maximum(0, Z)

    def forward(self, X):
        Z = np.dot(self.W, X) + self.b
        return self.relu(Z)

    def JacMx(self, x, v):
        z = np.dot(self.W, x) + self.b
        dReLU_dZ = z > 0
        return np.diag(dReLU_dZ.flatten()) @ self.W @ v
    
    def JacTMx(self, X, V):
        # Jacobian of ReLU with respect to X times V
        Z = np.dot(self.W, X) + self.b
        dReLU_dZ = Z > 0
        return np.dot(self.W.T, dReLU_dZ * V)

    def JacTMW(self, X, V):
        # Jacobian of ReLU with respect to W times V
        Z = np.dot(self.W, X) + self.b
        dReLU_dZ = Z > 0
        return np.dot(dReLU_dZ * V, X.T)

    def JacTMb(self, X, V):
        # Jacobian of ReLU with respect to b times V
        Z = np.dot(self.W, X) + self.b
        dReLU_dZ = Z > 0
        return dReLU_dZ * V
    
    def JacobianTest(self, x):
        d = np.random.randn(*x.shape)
        d = d / np.linalg.norm(d)
        eps = 1 
        E = []
        E2 = []

        for _ in range(10):
            y1 = np.linalg.norm(self.forward(x + eps * d) - self.forward(x))
            y2 = np.linalg.norm(self.forward(x + eps * d) - self.forward(x) - self.JacMx(x, eps * d))
            E.append(y1)
            E2.append(y2)
            eps = eps * 0.5
        
        plt.plot(E, label='first order error')
        plt.plot(E2, label='second order error')
        plt.yscale('log')
        plt.legend()
        plt.show()



class SoftmaxLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)

    def softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return e_Z / np.sum(e_Z, axis=0, keepdims=True)

    def forward(self, X):
        Z = np.dot(self.W, X) + self.b
        A = self.softmax(Z)
        return A
        
if __name__ == '__main__':
    l = ReLULayer(3, 3)
    X = np.array([[1,2,3]]).T
    C = np.array([[0, 0, 1, 0]]).T