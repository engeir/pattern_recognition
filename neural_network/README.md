# Neural network

#### Set up

To install all needed packages, run

```bash
pip install -r requirements.txt
```



#### How it works

- Initializing
  
  The neural network is a class initiated by giving the number of hidden layer `L` (i.e. not including input and output layer). The variable `layer_dim` must be a list of integers describing the number of neurons in each hidden layer and in the output layer. `i_num` is the number of dimensions of the input parameters. An optional argument can be given to specify the kernel function that should be used in the parameter `kernel_func`, where the default is the logistic function and the only other option is the rectified linear unit (ReLU).

- Training
  
  The network can be trained by calling the method `training` with the input and output vectors as required arguments, in addition to the dimension of the input space as an int. An optional argument can be given to accept a momentum parameter.
  
  The method returns the epoch error for the training, and several rounds of training may be needed.

- Usage
  
  To apply the neural network to a new set of data you call the method `propagate_forward`, where the only required argument is the test data set.



#### Example of usage

Examples of how to use the neural network can be found on the `main*.py` files. A short version from `main40.py` is presented below:

```python
import numpy as np
import backpropagation as bp

# The goal is to train the network to fit the function below on the domain x in [0, 10].
y = lambda x: 0.3 + 0.2 * np.cos(2 * np.pi * x)
xtr = np.linspace(0, 1, 1000)
ytr = y(xtr)
xte = np.linspace(0, 1, 50)

L = 2
i_num = 1
nn = bp.NeuralNetwork(L=L, layer_dim=[1, 4, 1], i_num=i_num, kernel_func='logistic')
nn.set_error_const(0.004)
nn.set_momentum(0.0015)

accuracy = .5
while accuracy < .9:
    accuracy = nn.training(xtr, ytr, i_num, momentum=True)
    print(f'{accuracy}', end='\r')

yte = nn.propagate_forward(xte)
```


