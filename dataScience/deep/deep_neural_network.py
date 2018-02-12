import numpy as np

class neural_network(object):

    def __init__(self):
        self.parameters={}

    @staticmethod
    def sigmoid(z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- sigmoid(z)
        """

        s = 1 / (1 + np.exp(-z))

        return s

    @staticmethod
    def relu(z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.

        Return:
        s -- relu(z)
        """

        z[z < 0] = 0

        return z

    def init_parm_deep(self, layer_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each laye
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", .
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[
        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(3)
        # parameters = {}
        L = len(layer_dims)  # number of layers in the network
        for l in range(1, L):
            self.parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])
            self.parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (self.parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert (self.parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return self.parameters

    def linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.
        Arguments:
        A -- activations from previous layer (or input data): (size of previous
        W -- weights matrix: numpy array of shape (size of current layer, size o
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        Returns:
        Z -- the input of the activation function, also called pre-activation pa
        cache -- a python dictionary containing "A", "W" and "b" ; stored for co
        """
        Z = np.dot(W, A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer
        Arguments:
        A_prev -- activations from previous layer (or input data): (size of prev
        W -- weights matrix: numpy array of shape (size of current layer, size o
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text
        Returns:
        A -- the output of the activation function, also called the post-activat
        cache -- a python dictionary containing "linear_cache" and "activation_c
        stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache