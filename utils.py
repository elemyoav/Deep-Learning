import numpy as np
import matplotlib.pyplot as plt

def col_mean(M):
    return np.mean(M, axis=1, keepdims=True)


def SGD(Xt, Yt, Xv, Yv, NN, lr, batch_size=64, epochs=200):
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    # fix batch size if it is not < number of samples
    if batch_size >= Xt.shape[1]:
        batch_size = Xt.shape[1]

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

def plot_data(NN, Xt, Yt, Xv, Yv):
    # plot the points of Xt colored by their class
    outputT = NN.forward(Xt)
    # plot the points of Xv colored by their class
    outputV = NN.forward(Xv)

    plt.subplot(2, 2, 1)  # 1 row, 3 columns, first subplot
    plt.title('Training True')
    plt.scatter(Xt[0], Xt[1], c=np.argmax(Yt, axis=0))

    # plot the points of Xt colored by their class
    plt.subplot(2, 2, 2)  # 1 row, 3 columns, second subplot
    plt.title('Training Prediction')
    plt.scatter(Xt[0], Xt[1], c=np.argmax(outputT, axis=1))

    # plot the points of Xv colored by their class
    plt.subplot(2, 2, 3)  # 1 row, 3 columns, third subplot
    plt.title('Validation True')
    plt.scatter(Xv[0], Xv[1], c=np.argmax(Yv, axis=0))

    plt.subplot(2, 2, 4)  # 1 row, 3 columns, fourth subplot
    plt.title('Validation Prediction')
    plt.scatter(Xv[0], Xv[1], c=np.argmax(outputV, axis=1))

    plt.show()

def plot_loss_and_accuracy(tloss, taccuracy, vloss, vaccuracy):
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    plt.plot(tloss)

    plt.subplot(2, 2, 2)
    plt.title('Training Accuracy')
    plt.plot(taccuracy)

    plt.subplot(2, 2, 3)
    plt.title('Validation Loss')
    plt.plot(vloss)

    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    plt.plot(vaccuracy)

    plt.show()


def Gradient_test(loss_layer, x, y):
    grad = loss_layer.grad_Θ(x, y)
    d = np.random.randn(grad.shape[0], 1)
    d = d / np.linalg.norm(d)
    eps = 1.
    E = []
    E2 = []
    for _ in range(30):
        y1 = np.linalg.norm(loss_layer.loss_Θ(x, y, dΘ = eps * d) - loss_layer.loss_Θ(x, y))
        y2 = np.linalg.norm(loss_layer.loss_Θ(x, y, dΘ = eps * d) - loss_layer.loss_Θ(x, y) - np.dot(eps * d.T, grad))
        E.append(y1)
        E2.append(y2)
        eps = eps * 0.5
    
    plt.plot(E, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def JacobianTest(layer, x, residual=False):
    
    if residual:
        dim = layer.W1.size + layer.W2.size + layer.b1.size + layer.b2.size + x.size
    else:
        dim = layer.W.size + layer.b.size + x.size

    d = np.random.randn(dim, 1)
    d = d / np.linalg.norm(d)
        
    eps = 1.
    E = []
    E2 = []

    for _ in range(30):
        y1 = np.linalg.norm(layer.forward_Θ(x, Θ=eps * d) - layer.forward_Θ(x))
        y2 = np.linalg.norm(layer.forward_Θ(x, Θ=eps * d) - layer.forward_Θ(x) - layer.JacΘMv(x, eps * d))
        E.append(y1)
        E2.append(y2)
        eps = eps * 0.5

    plt.plot(E, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def JacobianTransposeTest(layer, x):
    dimv = layer.W.size + layer.b.size + x.size
    dimu = layer.W.shape[0]
    v = np.random.randn(dimv, 1)
    u = np.random.randn(dimu, 1)

    j1 = u.T @ layer.JacΘMv(x, v)
    j2 = v.T @ layer.JacΘTMv(x, u)

    passed = np.abs(j1 - j2) < 1e-10
    return j1, j2, passed

def add_row_of_ones(X):
    return np.vstack((X, np.ones(X.shape[1])))