import numpy as np
import matplotlib.pyplot as plt

def col_mean(M):
    """
    Computes the column-wise mean of a given matrix.

    This function calculates the mean of each row in the input matrix M, effectively reducing the dimensionality of M from (n, m) to (n, 1). Each element in the output matrix represents the average of a row from the input matrix.

    Parameters:
    M (numpy.ndarray): A matrix of size (n, m) where 'n' is the number of rows and 'm' is the number of columns.

    Returns:
    numpy.ndarray: A column vector of size (n, 1), where each element is the mean of the corresponding row in the input matrix M.

    Example:
    >>> M = np.array([[1, 2, 3],
                      [4, 5, 6]])
    >>> col_mean(M)
    array([[2],
           [5]])
    """
    return np.mean(M, axis=1, keepdims=True)


def SGD(Xt, Yt, Xv, Yv, NN, lr, batch_size=64, epochs=200):
    """
    Preforms mini-batch SGD on the given neural network.
    
    Parameters:
    Xt (numpy.ndarray): Training data matrix of size (n, m) where 'n' is the number of features and 'm' is the number of training samples.
    Yt (numpy.ndarray): Training labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of training samples.
    Xv (numpy.ndarray): Validation data matrix of size (n, m) where 'n' is the number of features and 'm' is the number of validation samples.
    Yv (numpy.ndarray): Validation labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of validation samples.
    NN (NeuralNetwork): The neural network to train.
    lr (float): The learning rate.
    batch_size (int): The batch size.
    epochs (int): The number of epochs.

    Returns:
    training_loss (list): A list of the training loss at each epoch.
    validation_loss (list): A list of the validation loss at each epoch.
    training_accuracy (list): A list of the training accuracy at each epoch.
    validation_accuracy (list): A list of the validation accuracy at each epoch.
    """

    # Initialize lists to store the loss and accuracy at each epoch
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
            # Backpropagation and update parameters
            gradients = NN.backpropagation(Xb, Yb)       
            NN.update_parameters(gradients, lr)

        # Compute the loss and accuracy on the training set
        loss = NN.loss(Xt, Yt)
        accuracy = NN.accuracy(Xt, Yt)
        print(f'Epoch {epoch}, training loss: {loss}')# TODO: comment this out
        print(f'Epoch {epoch}, training accuracy: {accuracy}')# TODO: comment this out
        training_loss.append(loss)
        training_accuracy.append(accuracy)

        # Compute the loss and accuracy on the validation set
        loss = NN.loss(Xv, Yv)
        accuracy = NN.accuracy(Xv, Yv)
        print(f'Epoch {epoch}, validation loss: {loss}')# TODO: comment this out
        print(f'Epoch {epoch}, validation accuracy: {accuracy}')# TODO: comment this out
        validation_loss.append(loss)
        validation_accuracy.append(accuracy)

    return training_loss, validation_loss, training_accuracy, validation_accuracy

def plot_data(NN, Xt, Yt, Xv, Yv):
    """
    Plots the data and the predictions of the given neural network.
    This function assumes that our data is 2-dimensional and our neural network is a classifier.

    Parameters:
    NN (NeuralNetwork): The neural network to plot.
    Xt (numpy.ndarray): Training data matrix of size (2, m) where 2 is the number of features and 'm' is the number of training samples.
    Yt (numpy.ndarray): Training labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of training samples.
    Xv (numpy.ndarray): Validation data matrix of size (2, m) where 2 is the number of features and 'm' is the number of validation samples.
    Yv (numpy.ndarray): Validation labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of validation samples.

    Returns:
    None
    """

    # Compute the predictions of the neural network on the training and validation sets
    prediction_training = NN.forward(Xt)
    prediction_validation = NN.forward(Xv)

    plt.figure(figsize=(10, 8))
     
    # plot the points of Xt colored by their true class
    plt.subplot(2, 2, 1)
    plt.title('Training True')
    plt.scatter(Xt[0], Xt[1], c=np.argmax(Yt, axis=0))

    # plot the points of Xt colored by their predicted class
    plt.subplot(2, 2, 2)
    plt.title('Training Prediction')
    plt.scatter(Xt[0], Xt[1], c=np.argmax(prediction_training, axis=1))

    # plot the points of Xv colored by their true class
    plt.subplot(2, 2, 3)
    plt.title('Validation True')
    plt.scatter(Xv[0], Xv[1], c=np.argmax(Yv, axis=0))

    # plot the points of Xv colored by their predicted class
    plt.subplot(2, 2, 4)
    plt.title('Validation Prediction')
    plt.scatter(Xv[0], Xv[1], c=np.argmax(prediction_validation, axis=1))

    plt.show()

def plot_loss_and_accuracy(tloss, taccuracy, vloss, vaccuracy):
    """
    Plots the loss and accuracy of the training and validation sets.

    Parameters:
    tloss (list): A list of the training loss at each epoch.
    taccuracy (list): A list of the training accuracy at each epoch.
    vloss (list): A list of the validation loss at each epoch.
    vaccuracy (list): A list of the validation accuracy at each epoch.

    Returns:
    None
    """

    plt.figure(figsize=(10, 8))
    
    # Plot the loss on the training set
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    plt.plot(tloss)

    # Plot the accuracy on the training set
    plt.subplot(2, 2, 2)
    plt.title('Training Accuracy')
    plt.plot(taccuracy)

    # Plot the loss on the validation set
    plt.subplot(2, 2, 3)
    plt.title('Validation Loss')
    plt.plot(vloss)

    # Plot the accuracy on the validation set
    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    plt.plot(vaccuracy)

    plt.show()


def Gradient_test(loss_layer, x, y):
    """
    Plots the results of the gradient test presented in class in log scale.
    the method plots 2 lines, one for the first order error and one for the second order error.
    we expect the first order error to decrease linearly and the second order error to decrease quadratically

    Parameters:
    loss_layer (LossLayer): The loss layer to test, must have the methods loss_Θ and grad_Θ.
    where grad_Θ is the vectorized gradient with respect to the layer's parameters and x.
    loss_Θ is a method to compute the loss in the direction of dΘ, meaning it nudges the parameters (Θ) by dΘ and computes the loss.
    x (numpy.ndarray): The input data.
    y (numpy.ndarray): The labels.
    """

    # first we compute the gradient with respect to the parameters
    grad = loss_layer.grad_Θ(x, y)

    # d would be a random unit vector
    d = np.random.randn(grad.shape[0], 1)
    d = d / np.linalg.norm(d)

    # we start with a large epsilon and decrease it by half each iteration
    eps = 1.

    # we store the errors in these lists
    E = []
    E2 = []
    for _ in range(30):
        # compute || loss(Θ + eps * d) - loss(Θ) ||, expected to be O(ε)
        e1 = np.linalg.norm(loss_layer.loss_Θ(x, y, dΘ = eps * d) - loss_layer.loss_Θ(x, y))

        # compute || loss(Θ + eps * d) - (loss(Θ) + eps * d.T @ grad) ||, expected to b O(ε^2)
        e2 = np.linalg.norm(loss_layer.loss_Θ(x, y, dΘ = eps * d) - loss_layer.loss_Θ(x, y) - np.dot(eps * d.T, grad))

        E.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    # plot the results in log scale
    plt.plot(E, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def JacobianTest(layer, x, residual=False):
    """
    Plots the results of the Jacobian test presented in class in log scale.
    the method plots 2 lines, one for the first order error and one for the second order error.
    we expect the first order error to decrease linearly and the second order error to decrease quadratically

    Parameters:
    layer (Layer): The layer to test, must have the methods JacΘMv and forward_Θ.
    where JacΘMv is the Jacobian of the layer with respect to Θ times a vector v,
    and forward_Θ is the forward pass of the layer nudged by Θ.
    x (numpy.ndarray): The input data.
    residual (bool): Whether the layer is a residual layer or not.

    Returns:
    None
    """

    # set the dimensions of d according to the type of layer
    if residual:
        dim = layer.W1.size + layer.W2.size + layer.b1.size + layer.b2.size + x.size
    else:
        dim = layer.W.size + layer.b.size + x.size

    # d would be a random unit vector
    d = np.random.randn(dim, 1)
    d = d / np.linalg.norm(d)
    
    # we start with a large epsilon and decrease it by half each iteration
    eps = 1.

    # we store the errors in these lists
    E = []
    E2 = []

    for _ in range(30):
        # compute || f(Θ + eps * d) - f(Θ) ||, expected to be O(ε)
        e1 = np.linalg.norm(layer.forward_Θ(x, Θ=eps * d) - layer.forward_Θ(x))

        # compute || f(Θ + eps * d) - (f(Θ) + JacΘMv * eps * d) ||, expected to b O(ε^2)
        e2 = np.linalg.norm(layer.forward_Θ(x, Θ=eps * d) - layer.forward_Θ(x) - layer.JacΘMv(x, eps * d))

        E.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    # plot the results in log scale
    plt.plot(E, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def JacobianTransposeTest(layer, x, residual=False):
    """
    This function tests the method JacΘTMv of the given layer.
    it assumes the layer has the methods JacΘMv and JacΘTMv,
    and that JacΘMv passed the JacobianTest.
    given that, it will preform the test as written in the notes and return the results and 
    if the test passed or not.

    Parameters:
    layer (Layer): The layer to test, must have the methods JacΘMv and JacΘTMv.
    where JacΘMv is the Jacobian of the layer with respect to Θ times a vector v,
    and JacΘTMv is the Jacobian transpose of the layer with respect to Θ times a vector v.
    x (numpy.ndarray): The input data.
    residual (bool): Whether the layer is a residual layer or not.

    Returns:
    j1 (numpy.ndarray): The result of u.T @ JacΘ @ v.
    j2 (numpy.ndarray): The result of v.T @ JacΘ @ u.
    passed (bool): Whether the test passed or not (meaning the results are close enough).
    """

    # set the dimensions of d according to the type of layer
    if residual:
        dimv = layer.W1.size + layer.W2.size + layer.b1.size + layer.b2.size + x.size
        dimu = layer.W2.shape[0]
    else:
        dimv = layer.W.size + layer.b.size + x.size
        dimu = layer.W.shape[0]
    
    # initialize random vectors u and v
    v = np.random.randn(dimv, 1)
    u = np.random.randn(dimu, 1)

    j1 = u.T @ layer.JacΘMv(x, v)
    j2 = v.T @ layer.JacΘTMv(x, u)

    passed = np.abs(j1 - j2) < 1e-10
    return j1, j2, passed


def NetworkGradientTest(N, x, y):
    """
    Plots the results of the network gradient test presented in class in log scale.
    the method plots 2 lines, one for the first order error and one for the second order error.
    we expect the first order error to decrease linearly and the second order error to decrease quadratically

    Parameters:
    N (NeuralNetwork): The network to test, must have the methods grad_Θ and loss_Θ.
    where grad_Θ is the vectorized gradient with respect to the network's parameters and x.
    loss_Θ is a method to compute the loss in the direction of dΘ, meaning it nudges the parameters (Θ) by dΘ and computes the loss.
    x (numpy.ndarray): The input data.
    y (numpy.ndarray): The labels.

    Returns:
    None
    """

    # compute the gradient with respect to the parameters
    grad = N.grad_Θ(x, y)

    # d would be a random unit vector
    d = np.random.randn(grad.shape[0], 1)
    d = d / np.linalg.norm(d)

    # we start with a large epsilon and decrease it by half each iteration
    eps = 1.

    # we store the errors in these lists
    E = []
    E2 = []


    for _ in range(30):

        # compute || loss(Θ + eps * d) - loss(Θ) ||, expected to be O(ε)
        y1 = np.linalg.norm(N.loss_Θ(x, y, dΘ = eps * d) - N.loss_Θ(x, y))

        # compute || loss(Θ + eps * d) - (loss(Θ) + eps * d.T @ grad) ||, expected to b O(ε^2)
        y2 = np.linalg.norm(N.loss_Θ(x, y, dΘ = eps * d) - N.loss_Θ(x, y) - np.dot(eps * d.T, grad))

        E.append(y1)
        E2.append(y2)
        eps = eps * 0.5
    
    # plot the results in log scale
    plt.plot(E, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()