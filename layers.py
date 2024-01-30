import numpy as np
from matplotlib import pyplot as plt
from utils import col_mean, JacobianTest, JacobianTransposeTest


class HiddenLayer:
    """
    Hidden layer with activation function
    Used as a base class for other layers
    """

    def __init__(self, input_size, output_size, activation, dactivation):
        """
        input_size: number of input features
        output_size: number of output features
        activation: activation function
        dactivation: derivative of activation function
        """

        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
        self.activation = activation
        self.dactivation = dactivation

    def forward(self, X):
        """
        computes σ(WX + b)

        Parameters:
        X: input matrix of size (input_size, batch_size)

        Returns:
        output matrix of size (output_size, batch_size)
        """

        Z = np.dot(self.W, X) + self.b
        return self.activation(Z)

    def JacxMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to x times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size, 1)

        Returns:
        Jacobian of the activation function with respect to x times v
        """

        z = np.dot(self.W, x) + self.b
        dActivation_dZ = col_mean(self.dactivation(z))
        return (dActivation_dZ * self.W) @ v
    
    def JacWMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to W times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size * input_size, 1)

        Returns:
        Jacobian of the activation function with respect to W times v
        """

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
        """
        Computes the Jacobian of the activation function with respect to b times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size, 1)

        Returns:
        Jacobian of the activation function with respect to b times v
        """

        z = np.dot(self.W, x) + self.b
        dActivation_dZ = col_mean(self.dactivation(z))
        return dActivation_dZ * v

    def JacxTMv(self, X, V):
        """
        Computes the Jacobian of the activation function with respect to x transpose times v

        Parameters:
        X: input matrix of size (input_size, batch_size)
        V: matrix of size (output_size, batch_size)

        Returns:
        Jacobian of the activation function with respect to x transpose times v
        """

        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = col_mean(self.dactivation(Z))
        return np.dot(self.W.T, dActivation_dZ * V)

    def JacWTMv(self, X, V):
        """
        Computes the Jacobian of the activation function with respect to W transpose times v

        Parameters:
        X: input matrix of size (input_size, batch_size)
        V: matrix of size (output_size * input_size, batch_size)

        Returns:
        Jacobian of the activation function with respect to W transpose times v
        """

        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = self.dactivation(Z)
        return (dActivation_dZ * V) @ X.T / X.shape[1]

    def JacbTMv(self, X, V):
        """
        Computes the Jacobian of the activation function with respect to b transpose times v

        Parameters:
        X: input matrix of size (input_size, batch_size)
        V: matrix of size (output_size, batch_size)

        Returns:
        Jacobian of the activation function with respect to b transpose times v
        """

        Z = np.dot(self.W, X) + self.b
        dActivation_dZ = col_mean(self.dactivation(Z))
        return dActivation_dZ * V
    
    def unpack_Θ(self, dΘ):
        """
        Given a vector dΘ = [vec(dW), vec(db), dx].T
        unpacks it to dW, db, dx
        """

        dW = dΘ[:self.W.size].reshape(self.W.shape)
        db = dΘ[self.W.size:self.W.size + self.b.size].reshape(self.b.shape)
        dx = dΘ[self.W.size + self.b.size:].reshape(-1, 1)
        return dW, db, dx
    
    def forward_Θ(self, x, Θ=None):
        """
        Computes the result of activating the network on x when we nudge the
        weights and x by Θ

        Parameters:
        x: input vector of size (input_size, 1)
        Θ: the vectorized parameters to nudge, vector of size (output_size * input_size + output_size + input_size, 1)

        Returns:
        the result of activating the network on x when we nudge the weights and x by Θ
        """

        # if Θ is not provided, return the result of activating the network on x
        if Θ is None:
            return self.forward(x)
        
        # unpack Θ to dW, db, dx
        dW, db, dx = self.unpack_Θ(Θ)

        # nudge the weights and x
        self.W += dW
        self.b += db
        y = self.forward(x + dx)

        # reset the weights and x
        self.W -= dW
        self.b -= db

        # return the result of activating the network on x when we nudge the weights and x by Θ
        return y
    
    def JacΘMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to Θ times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size * input_size + output_size + input_size, 1)

        Returns:
        Jacobian of the activation function with respect to Θ times v
        """

        dWv = self.JacWMv(x, v[:self.W.size])
        dbv = self.JacbMv(x, v[self.W.size:self.W.size + self.b.size])
        dxv = self.JacxMv(x, v[self.W.size + self.b.size:])
        return dxv + dWv + dbv
    
    def JacΘTMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to Θ transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size * input_size + output_size + input_size, 1)

        Returns:
        Jacobian of the activation function with respect to Θ transpose times v
        """

        dWv = self.JacWTMv(x, v).reshape(-1, 1)
        dbv = self.JacbTMv(x, v)
        dxv = self.JacxTMv(x, v)

        return np.vstack((dWv, dbv, dxv))

    def update_weights(self, Θ, lr):
        """
        Updates the weights and biases of the layer

        Parameters:
        Θ: the vectorized parameters to nudge, vector of size (output_size * input_size + output_size, 1)
        lr: the learning rate

        Returns:
        None
        """

        # unpack Θ to dW, db
        dW, db = Θ
        dW = dW.reshape(self.W.shape)

        # update the weights and biases
        self.W -= lr * dW
        self.b -= lr * db

    def size(self):
        """
        Returns the number of parameters in the layer
        """

        return self.W.size + self.b.size

class HiddenResidualLayer:
    """
    Hidden layer with activation function
    Used as a base class for other layers
    """

    def __init__(self, input_size, output_size, activation, dactivation):
        self.W1 = np.random.randn(output_size, input_size)
        self.W2 = np.random.randn(input_size, output_size)
        self.b1 = np.random.randn(output_size, 1)
        self.b2 = np.random.randn(input_size, 1)
        self.activation = activation
        self.dactivation = dactivation

    def forward(self, X):
        """
        computes x + W2@σ(W1X + b1) + b2

        Parameters:
        X: input matrix of size (input_size, batch_size)

        Returns:
        output matrix of size (input_size, batch_size)
        """

        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.activation(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        return X + Z2
    
    def JacxMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to x times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (input_size, 1)

        Returns:
        Jacobian of the activation function with respect to x times v
        """

        da =col_mean(self.dactivation(np.dot(self.W1, x) + self.b1))
        return v + self.W2 @ (da *(self.W1 @ v))
    
    def JacW1Mv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to W1 times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size * input_size, 1)

        Returns:
        Jacobian of the activation function with respect to W1 times v
        """

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
        """
        Computes the Jacobian of the activation function with respect to W2 times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size * input_size, 1)

        Returns:
        Jacobian of the activation function with respect to W2 times v
        """

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
        """
        Computes the Jacobian of the activation function with respect to b1 transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size, 1)

        Returns:
        Jacobian of the activation function with respect to b1 transpose times v
        """

        da = col_mean(self.dactivation(np.dot(self.W1, x) + self.b1))
        return self.W2 @ (da * v)
    
    def Jacb2Mv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to b2 transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size, 1)

        Returns:
        Jacobian of the activation function with respect to b2 transpose times v
        """

        return v
    
    def JacxTMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to x transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (input_size, 1)

        Returns:
        Jacobian of the activation function with respect to x transpose times v
        """

        da = col_mean(self.dactivation(np.dot(self.W1, x) + self.b1))
        return v + self.W1.T @ (da * (self.W2.T @ v))
    
    def JacW1TMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to W1 transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size * input_size, 1)

        Returns:
        Jacobian of the activation function with respect to W1 transpose times v
        """

        da = self.dactivation(np.dot(self.W1, x) + self.b1)
        return da * (self.W2.T @ v) @ x.T
    
    def Jacb1TMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to b1 transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size, 1)

        Returns:
        Jacobian of the activation function with respect to b1 transpose times v
        """

        da = col_mean(self.dactivation(np.dot(self.W1, x) + self.b1))
        return da * (self.W2.T @ v)
    
    def JacW2TMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to W2 transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size * input_size, 1)

        Returns:
        Jacobian of the activation function with respect to W2 transpose times v
        """

        a = col_mean(self.activation(np.dot(self.W1, x) + self.b1))
        return v @ a.T
    
    def Jacb2TMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to b2 transpose times v

        Parameters:
        x: input vector of size (input_size, 1)
        v: vector of size (output_size, 1)

        Returns:
        Jacobian of the activation function with respect to b2 transpose times v
        """

        return v
    
    def unpack_Θ(self, dΘ):
        """
        Given a vector dΘ = [vec(dW1), vec(dW2), vec(db1), vec(db2), dx].T
        unpacks it to dW1, dW2, db1, db2, dx

        Parameters:
        dΘ: vector of size (output_size * input_size + input_size * output_size + output_size + input_size + input_size, 1)

        Returns:
        dW1: matrix of size (output_size, input_size)
        dW2: matrix of size (input_size, output_size)
        db1: vector of size (output_size, 1)
        db2: vector of size (input_size, 1)
        dx: vector of size (input_size, 1)
        """

        dW1 = dΘ[:self.W1.size].reshape(self.W1.shape)
        dW2 = dΘ[self.W1.size:self.W1.size + self.W2.size].reshape(self.W2.shape)
        db1 = dΘ[self.W1.size + self.W2.size:self.W1.size + self.W2.size + self.b1.size].reshape(self.b1.shape)
        db2 = dΘ[self.W1.size + self.W2.size + self.b1.size: self.W1.size + self.W2.size + self.b1.size + self.b2.size].reshape(self.b2.shape)
        dx = dΘ[self.W1.size + self.W2.size + self.b1.size + self.b2.size:].reshape(-1, 1)
        return dW1, dW2, db1, db2, dx

    def forward_Θ(self, x, Θ=None):
        """
        Computes the result of activating the network on x when we nudge the
        weights and x by Θ

        Parameters:
        x: input vector of size (input_size, 1)

        Returns:
        the result of activating the network on x when we nudge the weights and x by Θ
        """

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
        """
        Computes the Jacobian of the activation function with respect to Θ times v

        Parameters:
        x: input vector of size (input_size, 1) 
        v: vector of size (output_size * input_size + input_size * output_size + output_size + input_size + input_size, 1)

        Returns:
        Jacobian of the activation function with respect to Θ times v
        """

        dW1v = self.JacW1Mv(x, v[:self.W1.size])
        dW2v = self.JacW2Mv(x, v[self.W1.size:self.W1.size + self.W2.size])
        db1v = self.Jacb1Mv(x, v[self.W1.size + self.W2.size:self.W1.size + self.W2.size + self.b1.size])
        db2v = self.Jacb2Mv(x, v[self.W1.size + self.W2.size + self.b1.size:self.W1.size + self.W2.size + self.b1.size + self.b2.size])
        dxv = self.JacxMv(x, v[self.W1.size + self.W2.size + self.b1.size + self.b2.size:])
        return dW1v + dW2v + db1v + db2v + dxv
    
    def update_weights(self, Θ, lr):
        """
        Updates the weights and biases of the layer

        Parameters:
        Θ: the vectorized parameters to nudge, vector of size (output_size * input_size + input_size * output_size + output_size + input_size + input_size, 1)

        Returns:
        None
        """

        dW1, dW2, db1, db2 = Θ
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2
        self.b1 -= lr * db1
        self.b2 -= lr * db2
    
    def JacΘTMv(self, x, v):
        """
        Computes the Jacobian of the activation function with respect to Θ transpose times v

        Parameters:
        x: input vector of size (input_size, 1)

        Returns:
        Jacobian of the activation function with respect to Θ transpose times v
        """

        dW1v = self.JacW1TMv(x, v).reshape(-1, 1)
        dW2v = self.JacW2TMv(x, v).reshape(-1, 1)
        db1v = self.Jacb1TMv(x, v)
        db2v = self.Jacb2TMv(x, v)
        dxv = self.JacxTMv(x, v)

        return np.vstack((dW1v, dW2v, db1v, db2v, dxv))

    def size(self):
        """
        Returns the number of parameters in the layer
        """

        return self.W1.size + self.W2.size + self.b1.size + self.b2.size

class ReLULayer(HiddenLayer):
    """
    Hidden layer with ReLU activation function
    """

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.relu, self.drelu)
    
    def relu(self, Z):
        """
        computes ReLU(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        ReLU(Z)
        """
        return np.maximum(Z, 0)
    
    def drelu(self, Z):
        """
        computes the derivative of ReLU(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        the derivative of ReLU(Z)
        """

        return (Z > 0).astype(float)

class TanhLayer(HiddenLayer):
    """
    Hidden layer with tanh activation function
    """

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.tanh, self.dtanh)
    
    def tanh(self, Z):
        """
        computes tanh(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        tanh(Z)
        """

        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    def dtanh(self, Z):
        """
        computes the derivative of tanh(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        the derivative of tanh(Z)
        """
        
        return 1 - self.tanh(Z) ** 2

class ResidualTanhLayer(HiddenResidualLayer):
    """
    Hidden layer with tanh activation function
    """

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.tanh, self.dtanh)
    
    def tanh(self, Z):
        """
        computes tanh(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        tanh(Z)
        """
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    def dtanh(self, Z):
        """
        computes the derivative of tanh(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        the derivative of tanh(Z)
        """

        return 1 - self.tanh(Z) ** 2
    
class ResidualReLULayer(HiddenResidualLayer):
    """
    Hidden layer with ReLU activation function
    """

    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size, self.relu, self.drelu)
    
    def relu(self, Z):
        """
        computes ReLU(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        ReLU(Z)
        """

        return np.maximum(Z, 0)
    
    def drelu(self, Z):
        """
        computes the derivative of ReLU(Z)

        Parameters:
        Z: input matrix of size (output_size, batch_size)

        Returns:
        the derivative of ReLU(Z)
        """

        return (Z > 0).astype(float)
    
if __name__ == '__main__':
    l = TanhLayer(2, 4)
    x = np.array([[1, 2]]).T
    JacobianTest(l, x, residual=False)
    print(JacobianTransposeTest(l, x, residual=False))