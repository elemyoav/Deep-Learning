import numpy as np
from utils import col_mean, Gradient_test

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
    
    def grad_b(self, X, C):
        return None
    
    def loss(self, X, C):
        return np.sum(-np.log(self.forward(X)) * C.T) / X.shape[1]

    def update_weights(self, Θ, lr):
        dW, db = Θ
        self.W -= lr * dW

    def loss_Θ(self, X, C, dΘ=None):
        """
        recives a vector dΘ = [vec(dW), db].T
        and x
        unpacks them to dW, db
        and computes the loss in the direction of dΘ
        """
        if dΘ is None:
            return self.loss(X, C)
        
        dW, dx = self.unpack_Θ(dΘ)
        Z = np.dot(X.T + dx.T, self.W + dW)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return np.sum(-np.log(exp_Z / np.sum(exp_Z, axis=1, keepdims=True)) * C.T) / X.shape[1]
    
    def unpack_Θ(self, dΘ):
        dW = dΘ[:self.W.size].reshape(self.W.shape)
        dx = dΘ[self.W.size:].reshape(-1, 1)
        return dW, dx
    
    def grad_Θ(self, X, C):
        """
        vectorized gradient with respect to W and x
        grad_Θ(f) = [vec(grad_w(f)), grad_b(f)].T 
        """
        dW = self.grad_w(X, C)
        dx = self.grad_x(X, C)
        dΘ = np.concatenate((dW.reshape(-1, 1), dx), axis=0)
        return dΘ
    
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

class LinearLayer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
    
    def forward(self, x):
        return np.dot(self.W, x) + self.b

    def loss(self, x, y):
        y_pred = self.forward(x)
        return np.mean((y_pred - y) ** 2)
    
    def update_weights(self, Θ, lr):
        dW, db = Θ
        self.W -= lr * dW
        self.b -= lr * db

    def grad_w(self, x, y):
        y_pred = self.forward(x)
        return 2 * np.dot((y_pred - y), x.T)

    def grad_b(self, x, y):
        y_pred = self.forward(x)
        return 2 * np.mean(y_pred - y, axis=1, keepdims=True)

    def grad_x(self, x, y):
        y_pred = self.forward(x)
        return 2 * np.mean(np.dot(self.W.T, (y_pred - y)), axis=1, keepdims=True)
    

    def loss_Θ(self,x, y, dΘ=None):
        """d
        recives a vector dΘ = [vec(dW), db, vec(dx)].T
        and x
        unpacks them to dW, db, dx
        and computes the loss in the direction of dΘ
        """
        if dΘ is None:
            return self.loss(x, y)
        
        dW, db, dx = self.unpack_Θ(dΘ)
        y_pred = np.dot(self.W + dW, x + dx) + self.b + db
        return np.mean((y_pred - y) ** 2)
    
    def unpack_Θ(self, dΘ):
        dW = dΘ[:self.W.size].reshape(self.W.shape)
        db = dΘ[self.W.size:self.W.size + self.b.size].reshape(self.b.shape)
        dx = dΘ[self.W.size + self.b.size:].reshape(-1, 1)
        return dW, db, dx
    
    def grad_Θ(self, x, y):
        """
        vectorized gradient with respect to W and b
        grad_Θ(f) = [vec(grad_w(f)), grad_b(f), grad_x(f)].T 
        """

        dW = self.grad_w(x, y)
        db = self.grad_b(x, y)
        dx = self.grad_x(x, y)
        dΘ = np.concatenate((dW.reshape(-1, 1), db, dx), axis=0)
        return dΘ
    
    def grad_test_x(self, x, c):
        import matplotlib.pyplot as plt
        E = []
        E2 = []
        eps = 1.
        d = np.random.randn(x.shape[0], 1)
        d = d / np.linalg.norm(d)

        print(self.grad_x(x, c).shape)

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
    C = np.array([[1, 0, 0]]).T
    
    X = np.array([[1, 1, 1]]).T

    l = SoftmaxLayer(3, 3)

    Gradient_test(l, X, C)