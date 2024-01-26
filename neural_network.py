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
        gradients.append(dW)

        for i, layer in reversed(list(enumerate(self.layers))):
            dW = layer.JacTMW(self.cache[i], dx)

            db = layer.JacTMb(self.cache[i], dx)

            dx = layer.JacTMx(self.cache[i], dx)

            gradients.append((dW, db))
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients, lr):
        dW, gradients = gradients[-1], gradients[:-1]
        self.output_layer.W -= lr * dW

        for i, (dW, db) in enumerate(gradients):
           self.layers[i].b -= lr * db
           self.layers[i].W -= lr * dW


def SGD(Xt, Yt, Xv, Yv, NN, lr, batch_size=256, epochs=200):
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    for epoch in range(epochs):

        # Shuffle the data
        indices = np.arange(Xt.shape[1])
        np.random.shuffle(indices)
        Xt = Xt[:, indices]
        Yt = Yt[:, indices]

        # Mini-batch SGD
        for i in range(0, Xt.shape[1], batch_size):
            Xb = Xt[:, i:i+batch_size]
            Yb = Yt[:, i:i+batch_size]
            # Backpropagation
            gradients = NN.backpropagation(Xb, Yb)       

            # Update parameters
            NN.update_parameters(gradients, lr)

        # Compute loss
        loss = NN.loss(Xt, Yt)
        accuracy = NN.accuracy(Xt, Yt)
        print(f'Epoch {epoch}, training loss: {loss}')
        print(f'Epoch {epoch}, training accuracy: {accuracy}')
        training_loss.append(loss)
        training_accuracy.append(accuracy)
        loss = NN.loss(Xv, Yv)
        accuracy = NN.accuracy(Xv, Yv)
        print(f'Epoch {epoch}, validation loss: {loss}')
        print(f'Epoch {epoch}, validation accuracy: {accuracy}')
        validation_loss.append(loss)
        validation_accuracy.append(accuracy)

        

    return training_loss, validation_loss, training_accuracy, validation_accuracy

# Example usage with a small neural network
layer1 = ReLULayer(2, 128)  # Example sizes
layer2 = ReLULayer(128, 128)
loss_layer = SoftmaxLayer(128, 2)
NN = GenericNetwork(
    [
        layer1,
        layer2,
    ], 
    loss_layer
    )

if __name__ == '__main__':
    # Dummy data (replace with real data)
    swissroll = scipy.io.loadmat('HW1_Data(1)/SwissRollData.mat')
    Xt = swissroll['Yt']
    Yt = swissroll['Ct']
    Xv = swissroll['Yv']
    Yv = swissroll['Cv']

    # Train the network
    training_loss, validation_loss, training_accuracy, validation_accuracy = SGD(Xt, Yt, Xv, Yv, NN, lr=LR)

    # Plot the losses
    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.legend()
    plt.show()

    # Plot the accuracies
    plt.plot(training_accuracy, label='training accuracy')
    plt.plot(validation_accuracy, label='validation accuracy')
    plt.legend()
    plt.show()


    # plot the points of Xt colored by their class
    outputT = NN.forward(Xt)
    # plot the points of Xv colored by their class
    outputV = NN.forward(Xv)

    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
    plt.title('True')
    plt.scatter(Xt[0], Xt[1], c=Yt[1])

    # plot the points of Xt colored by their class
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
    plt.title('Training')
    plt.scatter(Xt[0], Xt[1], c=np.argmax(outputT, axis=1))

    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
    plt.title('Validation')
    plt.scatter(Xv[0], Xv[1], c=np.argmax(outputV, axis=1))

    plt.show()


