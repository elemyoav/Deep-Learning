import numpy as np
from matplotlib import pyplot as plt
from utils import col_mean, JacobianTest, JacobianTransposeTest


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
        return (dActivation_dZ * self.W) @ v
    
    def JacWMv(self, x, v):

        ##################################################
        # temporary fix since v is transposed for some reason
        dW = np.reshape(v, self.W.shape)
        dW = dW.T
        v = dW.reshape(-1, 1)
        #####################################################

        z = np.dot(self.W, x) + self.b
        dActivation_dZ = col_mean(self.dactivation(z))
        return np.diag(dActivation_dZ.flatten()) @ np.kron(x.T, np.eye(self.W.shape[0])) @ v
        
    def JacbMv(self, x, v):
        z = np.dot(self.W, x) + self.b
        dActivation_dZ = col_mean(self.dactivation(z))
        return dActivation_dZ * v

    def JacTxMv(self, X, V):
        # Jacobian of ReLU with respect to X times V
        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = col_mean(self.dactivation(Z))
        return np.dot(self.W.T, dActivation_dZ * V)

    def JacTWMv(self, X, V):
        # Jacobian of ReLU with respect to W times V
        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = self.dactivation(Z)
        return (dActivation_dZ * V) @ X.T / X.shape[1]

    def JacTbMv(self, X, V):
        # Jacobian of ReLU with respect to b times V
        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = col_mean(self.dactivation(Z))
        return dActivation_dZ * V
    
    def unpack_Θ(self, dΘ):
        dW = dΘ[:self.W.size].reshape(self.W.shape)
        db = dΘ[self.W.size:self.W.size + self.b.size].reshape(self.b.shape)
        dx = dΘ[self.W.size + self.b.size:].reshape(-1, 1)
        return dW, db, dx
    
    def forward_Θ(self, x, Θ=None):
        if Θ is None:
            return self.forward(x)
        dW, db, dx = self.unpack_Θ(Θ)
        self.W += dW
        self.b += db
        y = self.forward(x + dx)
        self.W -= dW
        self.b -= db
        return y
    
    def JacΘMv(self, x, v):
        dWv = self.JacWMv(x, v[:self.W.size])
        dbv = self.JacbMv(x, v[self.W.size:self.W.size + self.b.size])
        dxv = self.JacxMv(x, v[self.W.size + self.b.size:])
        return dxv + dWv + dbv
    
    def JacΘTMv(self, x, v):
        dWv = self.JacTWMv(x, v).reshape(-1, 1)
        dbv = self.JacTbMv(x, v)
        dxv = self.JacTxMv(x, v)

        return np.vstack((dWv, dbv, dxv))

    def update_weights(self, Θ, lr):
        dW, db = Θ
        dW = dW.reshape(self.W.shape)
        self.W -= lr * dW
        self.b -= lr * db


class HiddenResidualLayer:
    def __init__(self, input_size, output_size, activation, dactivation):
        self.W1 = np.random.randn(output_size, input_size)
        self.W2 = np.random.randn(input_size, output_size)
        self.b1 = np.random.randn(output_size, 1)
        self.b2 = np.random.randn(input_size, 1)
        self.activation = activation
        self.dactivation = dactivation

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.activation(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        return self.activation(X + Z2)
    
    def JacxMv(self, x, v):
        da =col_mean(self.dactivation(np.dot(self.W1, x) + self.b1)).flatten()
        J = self.W2 @ np.diag(da) @ self.W1 + np.eye(self.W2.shape[0])
        return J @ v
    
    def JacW1Mv(self, x, v):

        ##################################################
        # temporary fix since v is transposed for some reason
        dW = np.reshape(v, self.W1.shape)
        dW = dW.T
        v = dW.reshape(-1, 1)
        #####################################################

        da = col_mean(self.dactivation(np.dot(self.W1, x) + self.b1)).flatten()
        J = self.W2 @ np.diag(da) @ np.kron(x.T, np.eye(self.W1.shape[0]))
        return J @ v

    def JacW2Mv(self, x, v):

        ##################################################
        # temporary fix since v is transposed for some reason
        dW = np.reshape(v, self.W2.shape)
        dW = dW.T
        v = dW.reshape(-1, 1)
        #####################################################

        a = col_mean(self.activation(np.dot(self.W1, x) + self.b1))
        J = np.kron(a.T, np.eye(self.W2.shape[0]))
        return J @ v
    
    def Jacb1Mv(self, x, v):
        da = col_mean(self.dactivation(np.dot(self.W1, x) + self.b1)).flatten()
        J = self.W2 @ np.diag(da)
        return J @ v
    
    def Jacb2Mv(self, x, v):
        return v
    
    def JacTxMv(self, x, v):
        da = col_mean(self.dactivation(np.dot(self.W1, x) + self.b1)).flatten()
        return v + self.W1.T @ (da * (self.W2.T @ v))
    
    def JacTW1Mv(self, x, v):
        da = self.dactivation(np.dot(self.W1, x) + self.b1)
        return self.W2 @ (da * v) @ x.T
    
    def JacTb1Mv(self, x, v):
        da = col_mean(self.dactivation(np.dot(self.W1, x) + self.b1))
        return da * (self.W2.T @ v)
    
    def JacTW2Mv(self, x, v):#?
        a = self.activation(np.dot(self.W1, x) + self.b1)
        return np.reshape(np.kron(a, np.eye(self.W2.shape[0])) @ v, self.W2.shape)
    
    def JacTb2Mv(self, x, v):
        return v
    
    def unpack_Θ(self, dΘ):
        dW1 = dΘ[:self.W1.size].reshape(self.W1.shape)
        dW2 = dΘ[self.W1.size:self.W1.size + self.W2.size].reshape(self.W2.shape)
        db1 = dΘ[self.W1.size + self.W2.size:self.W1.size + self.W2.size + self.b1.size].reshape(self.b1.shape)
        db2 = dΘ[self.W1.size + self.W2.size + self.b1.size: self.W1.size + self.W2.size + self.b1.size + self.b2.size].reshape(self.b2.shape)
        dx = dΘ[self.W1.size + self.W2.size + self.b1.size + self.b2.size:].reshape(-1, 1)
        return dW1, dW2, db1, db2, dx

    def forward_Θ(self, x, Θ=None):
        if Θ is None:
            return self.forward(x)
        dW1, dW2, db1, db2, dx = self.unpack_Θ(Θ)
        self.W1 += dW1
        self.W2 += dW2
        self.b1 += db1
        self.b2 += db2
        y = self.forward(x + dx)
        self.W1 -= dW1
        self.W2 -= dW2
        self.b1 -= db1
        self.b2 -= db2
        return y
    
    def JacΘMv(self, x, v):
        dW1v = self.JacW1Mv(x, v[:self.W1.size])
        dW2v = self.JacW2Mv(x, v[self.W1.size:self.W1.size + self.W2.size])
        db1v = self.Jacb1Mv(x, v[self.W1.size + self.W2.size:self.W1.size + self.W2.size + self.b1.size])
        db2v = self.Jacb2Mv(x, v[self.W1.size + self.W2.size + self.b1.size:self.W1.size + self.W2.size + self.b1.size + self.b2.size])
        dxv = self.JacxMv(x, v[self.W1.size + self.W2.size + self.b1.size + self.b2.size:])
        return dW1v + dW2v + db1v + db2v + dxv
    
    def update_weights(self, Θ, lr):
        dW1, dW2, db1, db2 = Θ
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2
        self.b1 -= lr * db1
        self.b2 -= lr * db2

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

class ResidualTanhLayer(HiddenResidualLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.tanh, self.dtanh)
    
    def tanh(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    def dtanh(self, Z):
        return 1 - self.tanh(Z) ** 2
    
class ResidualReLULayer(HiddenResidualLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.relu, self.drelu)
    
    def relu(self, Z):
        return np.maximum(Z, 0)
    
    def drelu(self, Z):
        return (Z > 0).astype(float)
    
if __name__ == '__main__':
    l = ResidualTanhLayer(2, 4)
    x = np.array([[1, 2]]).T
    JacobianTest(l, x, residual=True)