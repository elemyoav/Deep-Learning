import numpy as np
from utils import col_mean, Gradient_test

class SoftmaxLayer:
    """
    Softmax layer, computes weighted softmax as learned in class.
    uses cross entropy loss as a loss function.
    """

    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)

    def forward(self, X):
        """
        computes softmax(X)
        
        Parameters:
        X is a matrix of size (input_size, batch_size)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        Z = np.dot(X.T, self.W)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def grad_x(self, X, C):
        """
        Computes the gradient of the loss with respect to x
        
        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a vector dx of size (input_size, 1)
        """

        return col_mean(np.dot(self.W, (self.forward(X) - C.T).T))
      
    def grad_w(self, X, C):
        """
        Computes the gradient of the loss with respect to W

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a matrix dW of size (input_size, output_size)
        """

        return np.dot(X, self.forward(X) - C.T)
    
    def grad_b(self, X, C):
        """
        Computes the gradient of the loss with respect to b

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a vector db of size (output_size, 1)
        """
        return None
    
    def loss(self, X, C):
        """
        Computes the cross entropy loss

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        the loss, a scalar
        """

        return np.sum(-np.log(self.forward(X)) * C.T) / X.shape[1]

    def update_weights(self, Θ, lr):
        """
        recives a vector dΘ = [vec(dW), db].T
        and updates W and b

        Parameters:
        Θ is a vector of size (input_size * output_size + output_size, 1)
        lr is the learning rate

        Returns:
        None
        """

        dW, _ = Θ
        self.W -= lr * dW

    def forward_Θ(self, x, Θ=None):
        """
        Computes softmax at the point x, when we move x and the weights W
        by the vector Θ

        Parameters:
        x is a vector of size (input_size, 1)
        Θ is a vector of size (input_size * output_size + input_size, 1)

        Returns:
        a vector of size (output_size, 1)
        """

        # in the case where we don't move x
        if Θ is None:
            return self.forward(x)
        
        # unpack Θ to dW, dx
        dW, dx = self.unpack_Θ(Θ)
        
        self.W += dW
        y = self.forward(x + dx)
        self.W -= dW

        return y
    
    def loss_Θ(self, X, C, dΘ=None):
        """
        Computes the loss at the point x, when we move x and the weights W
        by the vector dΘ

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)
        dΘ is a vector of size (input_size * output_size + input_size, 1)

        Returns:
        the loss, a scalar
        """

        # in the case where we don't move x
        if dΘ is None:
            return self.loss(X, C)
        
        # unpack dΘ to dW, dx
        dW, dx = self.unpack_Θ(dΘ)

        # compute the loss at the point x + dx, W + dW
        Z = np.dot(X.T + dx.T, self.W + dW)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return np.sum(-np.log(exp_Z / np.sum(exp_Z, axis=1, keepdims=True)) * C.T) / X.shape[1]
    
    def unpack_Θ(self, dΘ):
        """
        recives a vector dΘ = [vec(dW), dx].T
        and unpacks it to dW, dx

        Parameters:
        dΘ is a vector of size (input_size * output_size + input_size, 1)

        Returns:
        dW is a matrix of size (input_size, output_size)
        dx is a vector of size (input_size, 1)
        """

        dW = dΘ[:self.W.size].reshape(self.W.shape)
        dx = dΘ[self.W.size:].reshape(-1, 1)
        return dW, dx
    
    def grad_Θ(self, X, C):
        """
        vectorized gradient with respect to W and x
        grad_Θ(f) = [vec(grad_w(f)), grad_x(f)].T

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a vector dΘ of size (input_size * output_size + input_size, 1)
        """
        
        dW = self.grad_w(X, C)
        dx = self.grad_x(X, C)
        dΘ = np.concatenate((dW.reshape(-1, 1), dx), axis=0)
        return dΘ
    
    def size(self):
        """
        Returns the number of parameters in the layer
        """

        return self.W.size

class LinearLayer:
    """
    A Simple Linear Layer,
    computes Wx + b
    """

    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
    
    def forward(self, x):
        """
        computes Wx + b

        Parameters:
        x is a vector of size (input_size, 1)

        Returns:
        a vector of size (output_size, 1)
        """

        return np.dot(self.W, x) + self.b

    def loss(self, x, y):
        """
        computes the MSE loss

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)
        """

        y_pred = self.forward(x)
        return np.mean((y_pred - y) ** 2)
    
    def update_weights(self, Θ, lr):
        """
        recives a vector dΘ = [vec(dW), db, vec(dx)].T
        and updates W, b and x

        Parameters:
        Θ is a vector of size (input_size * output_size + output_size + input_size, 1)
        lr is the learning rate

        Returns:
        None
        """

        dW, db = Θ
        self.W -= lr * dW
        self.b -= lr * db

    def grad_w(self, x, y):
        """
        Computes the gradient of the loss with respect to W

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)

        Returns:
        a matrix dW of size (output_size, input_size)
        """

        y_pred = self.forward(x)
        return 2 * np.dot((y_pred - y), x.T)

    def grad_b(self, x, y):
        """
        Computes the gradient of the loss with respect to b

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)

        Returns:
        a vector db of size (output_size, 1)
        """

        y_pred = self.forward(x)
        return 2 * np.mean(y_pred - y, axis=1, keepdims=True)

    def grad_x(self, x, y):
        """
        Computes the gradient of the loss with respect to x

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)

        Returns:
        a vector dx of size (input_size, 1)
        """

        y_pred = self.forward(x)
        return 2 * np.mean(np.dot(self.W.T, (y_pred - y)), axis=1, keepdims=True)

    def loss_Θ(self,x, y, dΘ=None):
        """
        Computes the loss at the point x, when we move x, W and b
        by the vector dΘ

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)
        dΘ is a vector of size (input_size * output_size + output_size + input_size, 1)

        Returns:
        the loss, a scalar
        """

        if dΘ is None:
            return self.loss(x, y)
        
        dW, db, dx = self.unpack_Θ(dΘ)
        y_pred = np.dot(self.W + dW, x + dx) + self.b + db
        return np.mean((y_pred - y) ** 2)
    
    def unpack_Θ(self, dΘ):
        """
        recives a vector dΘ = [vec(dW), db, vec(dx)].T
        and unpacks it to dW, db, dx

        Parameters:
        dΘ is a vector of size (input_size * output_size + output_size + input_size, 1)

        Returns:
        dW is a matrix of size (output_size, input_size)
        db is a vector of size (output_size, 1)
        dx is a vector of size (input_size, 1)
        """

        dW = dΘ[:self.W.size].reshape(self.W.shape)
        db = dΘ[self.W.size:self.W.size + self.b.size].reshape(self.b.shape)
        dx = dΘ[self.W.size + self.b.size:].reshape(-1, 1)
        return dW, db, dx
    
    def grad_Θ(self, x, y):
        """
        vectorized gradient with respect to W, b and x
        grad_Θ(f) = [vec(grad_w(f)), grad_b(f), vec(grad_x(f))].T

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)

        Returns:
        a vector dΘ of size (input_size * output_size + output_size + input_size, 1)
        """

        dW = self.grad_w(x, y)
        db = self.grad_b(x, y)
        dx = self.grad_x(x, y)
        dΘ = np.concatenate((dW.reshape(-1, 1), db, dx), axis=0)
        return dΘ

    def size(self):
        """
        Returns the number of parameters in the layer
        """

        return self.W.size + self.b.size
    
if __name__ == '__main__':
    C = np.array([[1, 0, 0]]).T
    
    X = np.array([[1, 1, 1]]).T

    l = SoftmaxLayer(3, 3)

    Gradient_test(l, X, C)