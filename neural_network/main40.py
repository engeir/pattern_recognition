import random
import numpy as np
import matplotlib.pyplot as plt
import backpropagation as bp


random.seed(0)
y = lambda x: 0.3 + 0.2 * np.cos(2 * np.pi * x)


def prob_4_17():
    """Use a two-layer perceptron with a linear output unit to approximate the function
        y(x) = 0.3 + 0.2cos(2πx), x∈[0, 1]
    To this end, generate a sufficient number of data points from this function for the training.
    Use the backpropagation algorithm in one of its forms to train the network. In the sequel
    produce
        xte = 50
    more samples, feed them into the trained network, and plot the resulting outputs. How does
    it compare with the original curve? Repeat the procedure with different number of
    hidden units.
    """
    L = 2
    N = 1000
    M = 50
    xtr = np.linspace(0, 1, N)
    xte = np.linspace(0, 1, M)
    ytr = y(xtr)

    nn = bp.NeuralNetwork(
        L=L, layer_dim=[1, 4, 1], i_num=1, kernel_func='logistic')

    accuracy = [.5, .5]
    count = 0
    threshold = 20000
    percentage = 0.99
    difference = 0.001
    while accuracy[-1] < percentage or np.abs(accuracy[-1] - accuracy[-2]) > difference:
        accuracy.append(nn.training(xtr, ytr, 1, momentum=True))
        count += 1
        if count > threshold:
            print(f'No convergence after {count} epochs.')
            break
    if accuracy[-1] >= percentage:
        print('High enough accuracy.')
    else:
        print('Change is not big enough.')

    y_hat = nn.propagate_forward(xte)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(xtr, ytr)
    plt.plot(xte, y_hat)
    plt.legend(['Correct curve', 'Network curve'])
    plt.title('Curve y(x)')
    plt.subplot(1, 2, 2)
    plt.plot(accuracy)
    plt.title('Accuracy')
    plt.show()


prob_4_17()
