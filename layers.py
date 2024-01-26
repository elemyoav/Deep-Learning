import numpy as np
from matplotlib import pyplot as plt
from utils import col_mean


class HiddenLayer:
    def __init__(self, input_size, output_size, activation, dactivation):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
        self.activation = activation
        self.dactivation = dactivation

    def forward(self, X):
        Z = np.dot(self.W, X) + self.b
        return self.activation(Z)

    def JacxMv(self, x, v):
        z = np.dot(self.W, x) + self.b
        dActivation_dZ = col_mean(self.dactivation(z))
        J = np.diag(dActivation_dZ.flatten()) @ self.W
        return J @ v

    def JacTMx(self, X, V):
        # Jacobian of ReLU with respect to X times V
        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = col_mean(self.dactivation(Z))
        return np.dot(self.W.T, dActivation_dZ * V)

    def JacTMW(self, X, V):
        # Jacobian of ReLU with respect to W times V
        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = self.dactivation(Z)
        return np.dot(dActivation_dZ * V, X.T)

    def JacTMb(self, X, V):
        # Jacobian of ReLU with respect to b times V
        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = col_mean(self.dactivation(Z))
        return dActivation_dZ * V
    
    def JacobianTest(self, x):
        d = np.random.randn(*x.shape)
        d = d / np.linalg.norm(d)
        eps = 1.
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

    def update_weights(self, Θ, lr):
        dW, db = Θ
        self.W -= lr * dW
        self.b -= lr * db

class HiddenResidualLayer:
    
    def __init__(self, input_size, output_size, activation, dactivation):
        self.W1 = np.random.randn(output_size, input_size)
        self.W2 = np.random.randn(output_size, input_size)
        self.b1 = np.random.randn(output_size, 1)
        self.b2 = np.random.randn(output_size, 1)
        self.activation = activation
        self.dactivation = dactivation

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.activation(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        return X + Z2

    def JacxMv(self, x, v):
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.activation(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        dActivation_dZ1 = col_mean(self.dactivation(z1))
        J1 = np.diag(dActivation_dZ1.flatten()) @ self.W1
        return J1 @ v + self.W2 @ v

class ReLULayer(HiddenLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.relu, self.drelu)
    
    def relu(self, Z):
        return np.maximum(Z, 0)
    
    def drelu(self, Z):
        return (Z > 0).astype(float)

class TanhLayer(HiddenLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.tanh, self.dtanh)
    
    def tanh(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    def dtanh(self, Z):
        return 1 - self.tanh(Z) ** 2

if __name__ == '__main__':
    l = ReLULayer(3, 4)
    X = np.array([[1,2,3]]).T
    l.JacobianTest(X)