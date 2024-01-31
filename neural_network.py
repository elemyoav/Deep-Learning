import numpy as np
from layers import ReLULayer, TanhLayer, ResidualReLULayer, ResidualTanhLayer
from losses import SoftmaxLayer
import scipy.io
import matplotlib.pyplot as plt
from utils import plot_data, plot_loss_and_accuracy, SGD, NetworkGradientTest, NetworkJacobianTest
LR = 5e-8


class GenericNetwork:
    """
    Generic neural network class, can be used to create any fully connected neural network.

    Parameters:
    output_layer: the output layer of the network, should be either SoftmaxLayer or LinearLayer
    layers: a list of layers, each layer should be an extention of HiddenLayer
    """

    def __init__(self, output_layer, hidden_layers=[]):
        self.hidden_layers = hidden_layers
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

        Parameters:neural_network
        X is a matrix of size (input_size, batch_size)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        self.cache = [X]
        for layer in self.hidden_layers:
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
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            dW = layer.JacWTMv(self.cache[i], dx)

            db = layer.JacbTMv(self.cache[i], dx)

            dx = layer.JacxTMv(self.cache[i], dx)

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
            self.hidden_layers[i].update_weights(Θ, lr)

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
            return self.loss(x, y, clear_cache=False)
        
        self.cache = [x]
        start = 0

        # moves each layer by the corresponding part of dΘ
        for layer in self.hidden_layers:
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
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            grad = layer.JacΘTMv(self.cache[i], dx)
            grads.append(grad)
            dx = layer.JacxTMv(self.cache[i], dx)

        self.clear_cache()
        grads.reverse()
        return np.vstack(grads)
    
    def size(self):
        """
        Computes the total number of parameters in the network

        Returns:
        an integer
        """

        return sum([layer.size() for layer in self.hidden_layers]) + self.output_layer.size()

    def JacΘMv(self, x, v):
        """
        Computes the Jacobian of the forward function with respect to all of it's parameters
        multiplied by a vector v, that is computes J(x) * v

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a vector of size (total number of parameters, 1)

        Returns:
        a vector of size (output_size, 1)
        """

        # grads = []
        # self.forward_Θ(x)
        # grad = self.output_layer.grad_Θ(self.cache[-1], v)
        # dx 
        # grads.append(grad)
        # for i, layer in reversed(list(enumerate(self.hidden_layers))):
        #     grad = layer.JacΘTMv(self.cache[i], dx)
        #     grads.append(grad)
        #     dx = layer.JacxTMv(self.cache[i], dx)

        # self.clear_cache()
        # grads.reverse()
        # return np.vstack(grads)
        pass
    
    def forward_Θ(self, X, Θ=None):
        """
        Computes the output of the network on a given batch of data
        when nudging the parameters by Θ, that is we move all the parameters in the network

        Parameters:
        X is a matrix of size (input_size, batch_size)
        Θ is a vector of size (total number of parameters, 1)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        if Θ is None:
            return self.forward(X)
        
        self.cache = [X]
        start = 0

        # moves each layer by the corresponding part of Θ
        for layer in self.hidden_layers:
            end = start + layer.W.size + layer.b.size + X.size
            X = layer.forward_Θ(X, Θ[start:end])
            self.cache.append(X)
            start = end

        Y = self.output_layer.forward_Θ(X, Θ[start:])
        return Y
    
class ResidualNeuralNetwork:
    """
    Generic neural network class, can be used to create any fully connected neural network.
    """

    def __init__(self, output_layer, layers=[]):
        self.layers = layers
        self.output_layer = output_layer
        self.cache = []

    def loss(self, X, C, clear_cache=True):
        """
        Computes the loss of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)

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
        output = self.cache[-1]

        # compute the gradient of the loss with respect to the output of the network
        dx = self.output_layer.grad_x(output, Y)
        dW = self.output_layer.grad_w(output, Y)
        db = self.output_layer.grad_b(output, Y)

        # add the gradient of the output layer to the list of gradients
        gradients.append((dW, db))

        # backpropagate the gradient for each layer in reverse order
        for i, layer in reversed(list(enumerate(self.layers))):
            dW1 = layer.JacW1TMv(self.cache[i], dx)

            dW2 = layer.JacW2TMv(self.cache[i], dx)

            db1 = layer.Jacb1TMv(self.cache[i], dx)
            
            db2 = layer.Jacb2TMv(self.cache[i], dx)

            dx = layer.JacxTMv(self.cache[i], dx)

            gradients.append((dW1, dW2, db1, db2))
        
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
    
    def forward_Θ(self, X, Θ = None):
        """
        Computes the output of the network on a given batch of data
        when nudging the parameters by Θ, that is we move all the parameters in the network

        Parameters:
        X is a matrix of size (input_size, batch_size)
        Θ is a vector of size (total number of parameters, 1)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        if Θ is None:
            return self.forward(X)
        
        self.cache = [X]
        for layer in self.layers:
            X = layer.forward_Θ(X, Θ)
            self.cache.append(X)
        
        Y = self.output_layer.forward_Θ(X, Θ)
        return Y
    
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
            return self.loss(x, y, clear_cache=False)
        
        self.cache = [x]
        start = 0

        # moves each layer by the corresponding part of dΘ
        for layer in self.layers:
            end = start + layer.W1.size + layer.W2.size + layer.b1.size + layer.b2.size + x.size
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
            dx = layer.JacxTMv(self.cache[i], dx)

        self.clear_cache()
        grads.reverse()
        return np.vstack(grads)




if __name__ == '__main__':


    network = GenericNetwork(
                        hidden_layers=[
                                ReLULayer(2, 64), 
                                ReLULayer(64, 512),
                                ReLULayer(512, 1024),
                                ],
                        output_layer= SoftmaxLayer(1024, 2),
                        )
    swissroll = scipy.io.loadmat('HW1_Data(1)/SwissRollData.mat')
    Xt = swissroll['Yt']
    Yt = swissroll['Ct']
    Xv = swissroll['Yv']
    Yv = swissroll['Cv']

    print(network.loss(Xt, Yt))


