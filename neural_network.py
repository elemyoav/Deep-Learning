import numpy as np
from layers import ReLULayer
from losses import SoftmaxLayer
import scipy.io
import matplotlib.pyplot as plt
LR = 5e-5
class GenericNetwork:
    def __init__(self, layers, output_layer):
        self.layers = layers
        self.output_layer = output_layer
        self.cache = []

    def forward(self, X):
        self.cache = [X]
        for layer in self.layers:
            X = layer.forward(X)
            self.cache.append(X)
        
        Y = self.output_layer.forward(X)
        return Y

    def clear_cache(self):
        self.cache = []
    
    def backpropagation(self, Y):
        gradients = []
        output = self.cache[-1]
        dA = self.output_layer.grad_x(output, Y)
        dW = self.output_layer.grad_w(output, Y)
        self.output_layer.W -= LR * dW

        for i, layer in reversed(list(enumerate(self.layers))):
            dW = layer.JacTMW(self.cache[i], dA)
            db = layer.JacTMb(self.cache[i], dA)
            dX = layer.JacTMx(self.cache[i], dA)

            gradients.append((dW, db))
            dA = dX
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients, lr):
        for i, (dW, db) in enumerate(gradients):
            # iterate over the columns of db
            self.layers[i].b -= lr * db.sum(axis=1, keepdims=True)
            # iterate over the submatricies of dW,
            # that is dW=(dW1, dW2, ..., dWn)
            self.layers[i].W -= lr * dW


def SGD(Xt, Yt, Xv, Yv, NN, lr):
    losses = []
    for epoch in range(30):

        # Forward pass
        NN.forward(Xt)

        # Backpropagation
        gradients = NN.backpropagation(Yt)       

        # Update parameters
        NN.update_parameters(gradients, lr)

    return losses

# Example usage with a small neural network
layer1 = ReLULayer(2, 16)  # Example sizes
layer2 = ReLULayer(16, 64)
# layer3 = SoftmaxLayer(16, 2)
loss_layer = SoftmaxLayer(64, 2)
NN = GenericNetwork([layer1, layer2], loss_layer)

# Dummy data (replace with real data)
swissroll = scipy.io.loadmat('HW1_Data(1)/SwissRollData.mat')
Xt = swissroll['Yt']
Yt = swissroll['Ct']
Xv = swissroll['Yv']
Yv = swissroll['Cv']

# Train the network
losses = SGD(Xt, Yt, Xv, Yv, NN, lr=LR)

# # Plot the losses
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Over Time')
# plt.show()


# plot the points of Xt colored by their class
output = NN.forward(Xt)
# print(output.shape)
# exit()
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.scatter(Xt[0], Xt[1], c=Yt[0])

# plot the points of Xt colored by their class
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.scatter(Xt[0], Xt[1], c=np.argmax(output, axis=1))
plt.show()