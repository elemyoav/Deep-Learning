import numpy as np
from layers import ReLULayer, TanhLayer, ResidualReLULayer, ResidualTanhLayer
from losses import SoftmaxLayer
import scipy.io
import matplotlib.pyplot as plt
from utils import plot_data, plot_loss_and_accuracy, SGD, NetworkGradientTest
LR = 5e-5


class GenericNetwork:
    """
    Generic neural network class, can be used to create any fully connected neural network.

    Parameters:
    output_layer: the output layer of the network, should be either SoftmaxLayer or LinearLayer
    layers: a list of layers, each layer should be an extention of HiddenLayer
    """

    def __init__(self, output_layer, layers=[]):
        self.layers = layers
        self.output_layer = output_layer
        self.cache = [] # cache for forward pass

    def loss(self, X, C, clear_cache=True):
        """
        Computes the loss of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)
        clear_cache: if True, clears the cache after computing the loss

        Returns:
        a scalar loss
        """

        self.forward(X)
        loss = self.output_layer.loss(self.cache[-1], C)
        if clear_cache: self.clear_cache()
        return loss
    
    def accuracy(self, X, C):
        """
        Computes the accuracy of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a scalar accuracy
        """

        output = self.forward(X)
        self.clear_cache()
        return np.mean(np.argmax(output.T, axis=0) == np.argmax(C, axis=0))
    
    def forward(self, X):
        """
        Computes the output of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        self.cache = [X]
        for layer in self.layers:
            X = layer.forward(X)
            self.cache.append(X)
        
        Y = self.output_layer.forward(X)
        return Y

    def clear_cache(self):
        """
        Clears the cache of the network
        """

        self.cache = []
    
    def backpropagation(self, X, Y):
        """
        Computes the gradients of the network using backpropagation

        Parameters:
        X is a matrix of size (input_size, batch_size)
        Y is a matrix of size (output_size, batch_size)

        Returns:
        a list of gradients, each gradient is a tuple (dW, db)
        """

        # initialize the list of gradients
        gradients = []

        # preform a forward pass to fill the cache
        self.forward(X)

        # compute the gradient of the loss with respect to the output of the network
        output = self.cache[-1]
        dx = self.output_layer.grad_x(output, Y)
        dW = self.output_layer.grad_w(output, Y)
        db = self.output_layer.grad_b(output, Y)

        # add the gradient of the output layer to the list of gradients
        gradients.append((dW, db))

        # backpropagate the gradient for each layer in reverse order
        for i, layer in reversed(list(enumerate(self.layers))):
            dW = layer.JacTWMv(self.cache[i], dx)

            db = layer.JacTbMv(self.cache[i], dx)

            dx = layer.JacTxMv(self.cache[i], dx)

            # add the gradient of the layer to the list of gradients
            gradients.append((dW, db))

        # reverse the list of gradients and return it
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients, lr):
        """
        Updates the parameters of each layer of the network using the learning rate

        Parameters:
        gradients: a list of gradients, each gradient is a tuple (dW, db), corresponding to a layer
        lr: the learning rate

        Returns:
        None
        """

        Θ, gradients = gradients[-1], gradients[:-1]
        self.output_layer.update_weights(Θ, lr)

        for i, Θ in enumerate(gradients):
            self.layers[i].update_weights(Θ, lr)

    def loss_Θ(self, x, y, dΘ=None):
        """
        Computes the loss of the network on a given batch of data
        when nudging the parameters by dΘ, that is we move all the parameters in the network

        Parameters:
        x is a matrix of size (input_size, batch_size)
        y is a matrix of size (output_size, batch_size)
        dΘ is a vector of size (total number of parameters, 1), if None, we compute the loss without nudging the parameters

        Returns:
        a scalar loss
        """

        # in the case where we don't move the parameters
        if dΘ is None:
            return self.loss(x, y)
        
        self.cache = [x]
        start = 0

        # moves each layer by the corresponding part of dΘ
        for layer in self.layers:
            end = start + layer.W.size + layer.b.size + x.size
            x = layer.forward_Θ(x, dΘ[start:end])
            self.cache.append(x)
            start = end

        return self.output_layer.loss_Θ(x, y, dΘ[start:])
    
    def grad_Θ(self, x, y):
        """
        Computes the gradient of the loss of the loss function of the network
        with respect to all the parameters of the network, vecotrized.

        we use the same idea of backpropagation.

        Parameters:
        x is a matrix of size (input_size, batch_size)
        y is a matrix of size (output_size, batch_size)

        Returns:
        a vector of size (total number of parameters, 1)
        """
        
        grads = []
        self.loss_Θ(x, y)
        grad = self.output_layer.grad_Θ(self.cache[-1], y)
        dx = self.output_layer.grad_x(self.cache[-1], y)
        grads.append(grad)
        for i, layer in reversed(list(enumerate(self.layers))):
            grad = layer.JacΘTMv(self.cache[i], dx)
            grads.append(grad)
            dx = layer.JacTxMv(self.cache[i], dx)

        self.clear_cache()
        grads.reverse()
        return np.vstack(grads)


class ResidualNeuralNetwork:
    def __init__(self, output_layer, layers=[]):
        self.layers = layers
        self.output_layer = output_layer
        self.cache = []

    def loss(self, X, C):
        self.forward(X)
        loss = self.output_layer.loss(self.cache[-1], C)
        self.clear_cache()
        return loss
    
    def accuracy(self, X, C):
        output = self.forward(X)
        self.clear_cache()
        return np.mean(np.argmax(output.T, axis=0) == np.argmax(C, axis=0))
    
    def forward(self, X):
        self.cache = [X]
        for layer in self.layers:
            X = layer.forward(X)
            self.cache.append(X)
        
        Y = self.output_layer.forward(X)
        return Y

    def clear_cache(self):
        self.cache = []
    
    def backpropagation(self, X, Y):
        gradients = []

        output = self.forward(X)
        output = self.cache[-1]
        dx = self.output_layer.grad_x(output, Y)

        dW = self.output_layer.grad_w(output, Y)
        db = self.output_layer.grad_b(output, Y)
        gradients.append((dW, db))

        for i, layer in reversed(list(enumerate(self.layers))):
            dW1 = layer.JacTW1Mv(self.cache[i], dx)

            dW2 = layer.JacTW2Mv(self.cache[i], dx)

            db1 = layer.JacTb1Mv(self.cache[i], dx)
            
            db2 = layer.JacTb2Mv(self.cache[i], dx)

            dx = layer.JacTxMv(self.cache[i], dx)

            gradients.append((dW1, dW2, db1, db2))
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients, lr):
        Θ, gradients = gradients[-1], gradients[:-1]
        self.output_layer.update_weights(Θ, lr)

        for i, Θ in enumerate(gradients):
            self.layers[i].update_weights(Θ, lr)
    
    def forward_Θ(self, X, Θ):
        self.cache = [X]
        for layer in self.layers:
            X = layer.forward_Θ(X, Θ)
            self.cache.append(X)
        
        Y = self.output_layer.forward_Θ(X, Θ)
        return Y
    
    def JacΘMV(self, x, v):
        pass




if __name__ == '__main__':
    # Dummy data (replace with real data)
    # Example usage with a small neural network
    layer1 = ReLULayer(2, 16)  # Example sizes
    # layer2 = ResidualReLULayer(2, 128)
    # layer3 = ResidualReLULayer(2, 512)
    loss_layer = SoftmaxLayer(16, 2)

    NN = GenericNetwork(
        loss_layer,
        [
            layer1,
            # layer2,
            # layer3
        ]
    )
    swissroll = scipy.io.loadmat('HW1_Data(1)/SwissRollData.mat')
    Xt = swissroll['Yt']
    Yt = swissroll['Ct']
    Xv = swissroll['Yv']
    Yv = swissroll['Cv']

    x = Xt[:, 0].reshape(-1, 1)
    y = Yt[:, 0].reshape(-1, 1)
    NetworkGradientTest(NN, x, y)


