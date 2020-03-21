# Neural network

#### How it works

-   Initializing
    The neural network is a class initiated by giving the number of hidden layer `L` (i.e. not including input and output layer). The variable `layer_dim` must be a list of integers describing the number of neurons in each hidden layer and in the output layer. `i_num` is the number of input parameters / number of neurons in the input layer. An optional argument can be given to specify the kernel function that should be used in the parameter `kernel_func`, where the default is the logistic function and the only other option is the rectified linear unit (ReLU).

-   Training
    The network can be trained by calling the method `training` with the input and output vectors as required arguments, in addition to the dimension of the input space as an int. An optional argument can be given to accept a momentum parameter.

    The method returns the epoch error for the training, and several rounds of training may be needed.

-   Usage
    To apply the neural network to a new set of data you call the method `propagate_forward`, where the only required argument is the test data set.
