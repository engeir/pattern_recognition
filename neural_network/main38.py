import matplotlib.pyplot as plt
import numpy as np

import backpropagation as bp


def gaussian_training_vectors(features, means, sd, dim):
    """Make a set of N training feature vectors for M classes in a k dimensional space.

    Arguments:
        classes {int} -- how many classes the problem has
        features {int} -- how many training feature vectors you want per mean
        means {dict} -- a dict for each class containing a list of means for the class
        sd {np.array} -- a numpy array of every class' standard deviation with floating entries
        dim {int} -- the dimension of the space and hence the feature vectors

    Returns:
        3D np.array -- the set of training feature vectors in a 3D array
    """
    s = 0
    classes = 0
    for element in means:
        s += len(means[element])
    xtr = np.zeros((features * s, dim))
    ytr = np.zeros((features * s,))
    index = 0
    for element in means:
        for mean in range(len(means[element])):
            for _ in range(features):
                ytr[index] = classes
                xtr[index, :] = np.random.multivariate_normal(
                    means[element][mean], sd, 1)
                index += 1
        classes += 1
    return xtr, ytr


def logistic(x, a=1):
    """Calculate the output from the logistic function.

    Arguments:
        x {array or float or int} -- the function parameter as either a numpy array or as int/float
        a {float} -- defaults to 1; a constant describing the slope of the function (larger number gives steeper slope)

    Returns:
        array/float -- the value of the logistic function in the same format as the function parameter x
    """
    return 1 / (1 + np.exp(-a * x))


def d_logistic(x, a=1):
    """Find the derivative of the logistic function.

    Arguments:
        x {array or float or int} -- the function parameter as either a numpy array or as int/float
        a {float} -- defaults to 1; a constant describing the slope of the function (larger number gives steeper slope)

    Returns:
        array or float -- the value of the total derivative of the logistic function in the same format as the function parameter x
    """
    return a * logistic(x, a) * (1 - logistic(x, a))


def prob_4_2():
    """Using the computer, generate four 2D Gaussian random sequences with
        Σ=[0.01, 0.0; 0.0, 0.01],
        ω_1:
            μ_1 = [0, 0].T,
            μ_2 = [1, 1].T,
        ω_2:
            μ_3 = [0, 1].T,
            μ_4 = [1, 0].T.
    Produce 100 xtr from each distribution. Use the batch mode backpropagation algorithm (p. 170) of
    sec. 4.6 to train a two-layer perceptron with two hidden neurons and one in the output.
    Let the activation function be the logistic one with
        a = 1.
    Plot the error curve as a function of iteration steps. Experiment yourselves with various
    values of the learning parameter μ. Once the algorithm has converged, produce 50 more
    vectors from each distribution and try to classify them using the weights you have obtained.
    What is the percentage classification error?
    """
    features = 100  # number of feature vectors from each mean value/vector
    real_data = 50
    d = 2  # the dimension of the feature space
    means = {'1': [[0, 0], [1, 1]], '2': [[0, 1], [1, 0]]}
    sd = np.array([[0.01, 0.0], [0.0, 0.01]])
    xtr, ytr = gaussian_training_vectors(features, means, sd, d)
    xte, yte = gaussian_training_vectors(real_data, means, sd, d)

    L = 2

    # epoch_error = np.array([30, 28])
    accuracy = [.5, .5]

    nn = bp.NeuralNetwork(L=L, layer_dim=[2, 2, 1], i_num=2)

    count = 0
    while accuracy[-1] < 0.96 or np.abs(accuracy[-1] - accuracy[-2]) > 0.01:
        accuracy.append(nn.training(xtr, ytr, 2))
        count += 1
        if count > 10000:
            break

    print(f'Number of epochs before convergence: {count}')
    plot_surface(xte, yte, nn, accuracy)


def prob_4_10():
    means = {'1': [[0.4, 0.9], [2.0, 1.8], [2.3, 2.3], [2.6, 1.8]],
             '2': [[1.5, 1.0], [1.9, 1.0], [1.5, 3.0], [3.3, 2.6]]}
    features = 50
    d = 2
    sd = np.array([[0.008, 0.0], [0.0, 0.008]])
    xtr, ytr = gaussian_training_vectors(features, means, sd, d)

    L = 3
    layer_dim = [2, 3, 2, 1]

    epoch_error = np.array([30, 28])
    accuracy = [.5, .5]

    nn = bp.NeuralNetwork(L=L, layer_dim=layer_dim, i_num=2)
    nn.set_error_const(.006)

    count = 0
    threshold = 30000
    while accuracy[-1] < 0.96 or np.abs(accuracy[-1] - accuracy[-2]) > 0.01:
        # epoch_error = np.append(epoch_error, nn.training(xtr, ytr, 2)[0])
        accuracy.append(nn.training(xtr, ytr, 2))
        count += 1
        if count > threshold:
            print(f'No convergence after {count} epochs.')
            break

    if count < threshold + 1:
        print(f'Number of epochs before convergence: {count}')
    plot_surface(xtr, ytr, nn, accuracy)


def plot_surface(Xtr, Ytr, nn, accuracy):
    plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.semilogy(np.abs(epoch_error))
    plt.subplot(1, 2, 1)
    plt.title('Accuracy')
    plt.plot(accuracy)
    plt.subplot(1, 2, 2)
    plt.title('Classification surface')
    # Generate a grid of datapoints.
    x1min = np.min(Xtr[:, 0])
    x1max = np.max(Xtr[:, 0])
    x1margin = 0.05 * (x1max - x1min)

    x2min = np.min(Xtr[:, 1])
    x2max = np.max(Xtr[:, 1])
    x2margin = 0.05 * (x2max - x2min)

    x1axis = np.linspace(x1min - x1margin, x1max + x1margin, 200)
    x2axis = np.linspace(x2min - x2margin, x2max + x2margin, 200)

    X1grid = np.tile(x1axis, (len(x2axis), 1)).T.reshape(-1, 1)
    X2grid = np.tile(x2axis, (1, len(x1axis))).reshape(-1, 1)

    Xgrid = np.concatenate((X1grid, X2grid), axis=1)

    f_x = nn.propagate_forward(Xgrid)

    # Plot contour
    X1grid = X1grid.reshape(len(x2axis), -1)
    X2grid = X2grid.reshape(-1, len(x1axis))
    f_x = f_x.reshape(len(x2axis), len(x1axis))

    # Plot decision boundary and margins
    plt.contour(X1grid, X2grid, f_x, levels=[0], linestyles=(
        'solid'), linewidths=2, colors='k')

    plt.contourf(X1grid, X2grid, f_x, levels=np.linspace(
        np.min(f_x), np.max(f_x), 200), cmap='seismic')

    col = np.where(Ytr == 1.0, 'b', 'y')
    plt.scatter(Xtr[:, 0], Xtr[:, 1], c=col)

    plt.show()


np.random.seed(0)
# prob_4_2()
prob_4_10()
