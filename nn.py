import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Choose best layer sizes)
        self.layer_sizes = [3, 10, 2]  # input layer , hidden layer, output layer

        # Implemented FCNNs architecture here
        # There are 3 layers, between each two consecutive layers, there needs to be a weight matrix
        self.weight = [
            np.random.normal(size=(layer_sizes[1], layer_sizes[0])),  # weights between layer 0 and 1, aka W[0]
            np.random.normal(size=(layer_sizes[2], layer_sizes[1]))  # weights between layer 1 and 2, aka W[1]
        ]

        # Initialize bias to 0, for every layer.
        self.bias = [
            np.zeros((layer_sizes[1], 1)),  # bias vector between layer 0 and 1, aka B[0]
            np.zeros((layer_sizes[2], 1))  # bias vector between layer 1 and 2, aka B[1]
        ]
        pass

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """

        # Implemented activation function (sigmoid function) here.
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """

        # Implemented the forward function here.
        z = [
            np.zeros((self.layer_sizes[0], 1)),
            np.zeros((self.layer_sizes[1], 1)),
            np.zeros((self.layer_sizes[2], 1))
        ]

        z[0] = x

        for i in range(1, 3):
            # for each next layer, z is calculated as shown below
            z[i] = self.activation(self.weight[i - 1] @ z[i - 1] + self.bias[i - 1])

        return z[2]
