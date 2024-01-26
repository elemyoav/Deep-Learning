import numpy as np
from layers import ReLULayer
from losses import SoftmaxLayer
import scipy.io
import matplotlib.pyplot as plt
from utils import plot_data, plot_loss_and_accuracy, SGD
LR = 5e-5


class GenericNetwork:
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
            dW = layer.JacTMW(self.cache[i], dx)

            db = layer.JacTMb(self.cache[i], dx)

            dx = layer.JacTMx(self.cache[i], dx)

            gradients.append((dW, db))
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients, lr):
        Θ, gradients = gradients[-1], gradients[:-1]
        self.output_layer.update_weights(Θ, lr)

        for i, Θ in enumerate(gradients):
            self.layers[i].update_weights(Θ, lr)







if __name__ == '__main__':
    # Dummy data (replace with real data)
    # Example usage with a small neural network
    layer1 = ReLULayer(2, 128)  # Example sizes
    layer2 = ReLULayer(128, 128)
    loss_layer = SoftmaxLayer(128, 2)

    NN = GenericNetwork(
        loss_layer,
        [
            layer1,
            layer2,
        ]
    )
    swissroll = scipy.io.loadmat('HW1_Data(1)/SwissRollData.mat')
    Xt = swissroll['Yt']
    Yt = swissroll['Ct']
    Xv = swissroll['Yv']
    Yv = swissroll['Cv']

    # Train the network
    tloss, vloss, tacc, vacc = SGD(Xt, Yt, Xv, Yv, NN, lr=LR, epochs=50)

    plot_loss_and_accuracy(tloss, tacc, vloss, vacc)
    plot_data(NN, Xt, Yt, Xv, Yv)


