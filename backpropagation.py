import numpy as np


class NeuralNetwork:
    """Make a NeuralNetwork object that is able to first be trained and then used for classification problems."""

    # Problem 4.17:
    #   logistic:
    #     0.004
    #     0.0015
    #   rectified:
    #     ?
    #     ?
    ERROR_CONST = 0.004
    MOMENTUM = 0.0015

    def __init__(self, L, layer_dim, i_num, kernel_func=None):
        """Initiate the NeuralNetwork structure.

        Arguments:
            L {int} -- the amount of hidden layers you want
            layer_dim {list} -- a list of int describing how many neuron the hidden layers and the output layer should have
            i_num {int} -- the dimensionality of the input space
            kernel_func {str, optional} -- choose which type of kernel function you want to use
        """
        if (L + 1) != len(layer_dim):
            print('Please correctly specify how many neurons will be in each layer.')
            exit()
        if kernel_func == 'logistic':
            self.kernel_func = self.logistic
        elif kernel_func == 'rectified':
            self.kernel_func = self.rectified
        elif kernel_func is None:
            self.kernel_func = self.logistic
        else:
            print('You have a typo when specifying the kernel_func.')
            exit()
        self.L = L
        self.layer_dim = layer_dim
        self.i_num = i_num
        self.weights = []
        self.weights_correction = []

        self.make_weights()

    @staticmethod
    def rectified(v_value, d=True):
        if d:
            return 1 * (v_value > 0)
        return np.maximum(v_value, 0, v_value)

    @staticmethod
    def logistic(v_value, d=False):
        if d:
            return 1 / (1 + np.exp(- v_value)) * (1 - 1 / (1 + np.exp(- v_value)))
        return 1 / (1 + np.exp(- v_value))

    def make_weights(self):
        """Make weights for the L layers of the neural network.

        For each layer r, a matrix is put into the weights list with dimension (k_[r] × k_[r-1] + 1).
        """
        for i in range(self.L):
            self.weights.append(np.random.uniform(
                0, 1, (self.layer_dim[i + 1], self.layer_dim[i] + 1)))
            self.weights_correction.append(
                np.zeros((self.layer_dim[i + 1], self.layer_dim[i] + 1)))

    def propagate_forward(self, inputs, training_session=False):
        if training_session:
            v_list = []
            # The augmented inputs is set to be the first y's (i.e. output of layer 0).
            # dim y: (N × [feature space +1])
            y = inputs
            for i in range(self.L):
                if i == self.L - 1:
                    # dim y_hat: (N feature vectors)×([layer L])
                    v_L = self.weights[i] @ y.T
                    v_list.append(v_L.T)
                    y_hat = self.kernel_func(self.weights[i] @ y.T).T
                else:
                    # dim y: (N feature vectors)×([layer r] + 1)
                    vv = self.weights[i] @ y.T
                    v_list.append(vv.T)
                    v = (self.kernel_func(self.weights[i] @ y.T)).T
                    y = np.c_[v, np.ones(y.shape[0])]
            return v_list

        if inputs.shape == (len(inputs),):
            inputs = inputs.reshape((-1, 1))
        if inputs.shape[1] == self.i_num:
            inputs = np.c_[inputs, np.ones(inputs.shape[0])]
        y = inputs
        for i in range(self.L):
            if i == self.L - 1:
                # y_hat: dim(N feature vectors)×dim([layer L])
                v_L = self.weights[i] @ y.T
                y_hat = (self.kernel_func(self.weights[i] @ y.T)).T
            else:
                # y: dim(N feature vectors)×dim([layer r] + 1)
                vv = self.weights[i] @ y.T
                v = (self.kernel_func(self.weights[i] @ y.T)).T
                y = np.c_[v, np.ones(y.shape[0])]

        return y_hat

    def propagate_backward(self, inputs, ytr, momentum=False):
        # With momentum included, the weighs corrections from last epoch need to be kept. Or is it last

        ytr = ytr.reshape((len(ytr), 1))
        v_list = self.propagate_forward(inputs, training_session=True)

        # First step where we compare with the true class labels.
        accuracy = 1 - np.mean(np.abs(self.kernel_func(v_list[-1]) - ytr))
        delta = (self.kernel_func(v_list[-1]) - ytr) * \
            self.kernel_func(v_list[-1], d=True)
        if delta.shape[0] != inputs.shape[0]:
            # dim delta: (N × k_L)
            delta = delta.T
        y_prev = np.c_[self.kernel_func(v_list[-2]), np.ones(v_list[-2].shape[0])]
        if momentum:
            self.weights[-1] += self.MOMENTUM * self.weights_correction[-1] - \
                self.ERROR_CONST * (delta.T @ y_prev)
        else:
            self.weights[-1] -= self.ERROR_CONST * (delta.T @ y_prev)
        self.weights_correction[-1] -= self.ERROR_CONST * (delta.T @ y_prev)

        # delta_(r-1) = e_(r-1) * f'(v_(r-1))
        # e_(r-1) = sum_(k to k_r) delta_r * w_r
        # e has shape (N,k_[r-1]), weights has shape (k_[r], k_[r-1])
        if self.L > 2:
            for r in range(1, self.L - 1):
                e = delta @ self.weights[-r][:, :-1]
                delta = e * self.kernel_func(v_list[-(r + 1)], d=True)
                y_prev = np.c_[self.kernel_func(
                    v_list[-(r + 2)]), np.ones(v_list[-(r + 2)].shape[0])]
                if momentum:
                    self.weights[-(r + 1)] += self.MOMENTUM * self.weights_correction[-(r + 1)] - \
                        self.ERROR_CONST * (delta.T @ y_prev)
                else:
                    self.weights[-(r + 1)] -= self.ERROR_CONST * \
                        (delta.T @ y_prev)
                self.weights_correction[-(r + 1)
                                       ] -= self.ERROR_CONST * (delta.T @ y_prev)

        # Last step where we use the inputs (feature vectors) to correct the weights.
        e = delta @ self.weights[1][:, :-1]
        delta = e * self.kernel_func(v_list[0], d=True)
        if momentum:
            self.weights[0] += self.MOMENTUM * self.weights_correction[0] - \
                self.ERROR_CONST * (delta.T @ inputs)
        else:
            self.weights[0] -= self.ERROR_CONST * (delta.T @ inputs)
        self.weights_correction[0] -= self.ERROR_CONST * (delta.T @ inputs)

        return accuracy

    def training(self, xtr, ytr, dimension, momentum=False):
        if not isinstance(xtr, np.ndarray):
            print('The training set is not a numpy array.')
            exit()
        if xtr.shape[0] == dimension:
            xtr = xtr.T
        xtr = np.c_[xtr, np.ones(xtr.shape[0])]
        epoch_error = self.propagate_backward(
            xtr, ytr, momentum=momentum)

        return epoch_error
