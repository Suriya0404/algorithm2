import numpy as np

class neural_network(object):

    def __init__(self):
        self.parameters = {}

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
        Initialize the parameters.

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
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)
        return A, cache

    def L_model_forward(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()
        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
        every cache of linear_relu_forward() (there are L-1 of them,
        the cache of linear_sigmoid_forward() (there is one, indexed
        """

        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural
        # print(parameters)
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu')
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid')
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def compute_cost(AL, Y):
        """
        Implement the cost function defined by equation (7).
        Arguments:
        AL -- probability vector corresponding to your label predictions, shape
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat
        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]
        # Compute loss from aL and y.
        cost = -(1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y),np.log(1 - AL)))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we
        assert (cost.shape == ())
        return cost

    def linear_backward(dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current
        cache -- tuple of values (A_prev, W, b) coming from the forward propagati
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the pr
        dW -- Gradient of the cost with respect to W (current layer l), same shap
        db -- Gradient of the cost with respect to b (current layer l), same shap
        """

        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for c
        activation -- the activation to be used in this layer, stored as a text
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the p
        dW -- Gradient of the cost with respect to W (current layer l), same sha
        db -- Gradient of the cost with respect to b (current layer l), same sha
        """

        linear_cache, activation_cache = cache

        # if activation == "relu":
        #     dZ = relu_backward(dA, activation_cache)
        #     dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        # elif activation == "sigmoid":
        #     dZ = sigmoid_backward(dA, activation_cache)
        #     dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        if activation == "relu":
            dgZ = self.relu(activation_cache)
            dZ = np.multiply(dA, dgZ)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == "sigmoid":
            A = self.sigmoid(activation_cache)
            dgZ = np.multiply(A, 1 - A)
            dZ = np.multiply(dA, dgZ)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    ### ----> Check code below -

    def L_model_backward(AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LIN
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_for
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
        every cache of linear_activation_forward() with "relu" (it's
        the cache of linear_activation_forward() with "sigmoid" (it'
        Returns:
        grads -- A dictionary with the gradients
        grads["dA" + str(l)] = ...
        grads["dW" + str(l)] = ...
        grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
        # Initializing the backpropagation
        ### START CODE HERE ### (1 line of code)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        ### END CODE HERE ###
        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outp
        ### START CODE HERE ### (approx. 2 lines)
        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linea
        ### END CODE HERE ###
        for l in reversed(range(L - 1)):
            print('value of l is {}'.format(l))
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" +
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["d
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent
        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of L_model_
        Returns:
        parameters -- python dictionary containing your updated parameters
        parameters["W" + str(l)] = ...
        parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2  # number of layers in the neural network
        # Update rule for each parameter. Use a for loop.
        ### START CODE HERE ### (≈ 3 lines of code)
        for l in range(1, L):
            parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate
        ### END CODE HERE ###
        return parameters

    def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000):
        """
        Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (
        layers_dims -- dimensions of the layers (n_x, n_h, n_y)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- If set to True, this will print the cost every 100 iterati
        Returns:
        parameters -- a dictionary containing W1, W2, b1, and b2
        """

        np.random.seed(1)
        grads = {}
        costs = []  # to keep track of the cost
        m = X.shape[1]  # number of examples
        (n_x, n_h, n_y) = layers_dims
        # Initialize parameters dictionary, by calling one of the functions you
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = self.initialize_parameters(n_x, n_h, n_y)
        ### END CODE HERE ###
        # Get W1, b1, W2 and b2 from the dictionary parameters.
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs:
            ### START CODE HERE ### (≈ 2 lines of code)
            A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
            A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
            ### END CODE HERE ###
            # Compute cost
            ### START CODE HERE ### (≈ 1 line of code)
            cost = compute_cost(A2, Y)
            ### END CODE HERE ###
            # Initializing backward propagation
            dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
            # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1
            ### START CODE HERE ### (≈ 2 lines of code)
            dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
            dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
            ### END CODE HERE ###
            # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2,
            grads['dW1'] = dW1
            grads['db1'] = db1
            grads['dW2'] = dW2
            grads['db2'] = db2
            # Update parameters.
            ### START CODE HERE ### (approx. 1 line of code)
            parameters = update_parameters(parameters, grads, learning_rate)
            ### END CODE HERE ###
            # Retrieve W1, b1, W2, b2 from parameters
            W1 = parameters["W1"]
            b1 = parameters["b1"]
            W2 = parameters["W2"]
            b2 = parameters["b2"]

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if print_cost and i % 100 == 0:
                costs.append(cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        return parameters

    def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000):
        """
        Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMO
        Arguments:
        X -- data, numpy array of shape (number of examples, num_px * num_px * 3
        Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (
        layers_dims -- list containing the input size and each layer size, of le
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop
        print_cost -- if True, it prints the cost every 100 steps
        Returns:
        parameters -- parameters learnt by the model. They can then be used to p
        """

        np.random.seed(1)
        costs = []  # keep track of cost
        # Parameters initialization.
        ### START CODE HERE ###
        parameters = initialize_parameters_deep(layers_dims)
        ### END CODE HERE ###
        # Loop (gradient descent)
        for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
            AL, caches = L_model_forward(X, parameters)
        ### END CODE HERE ###
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL, Y)
        ### END CODE HERE ###
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL, Y, caches)
        ### END CODE HERE ###
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        return parameters


if __name__ == '__main__':
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]
    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)
        print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T  # The
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    ### CONSTANTS DEFINING THE MODEL ####
    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)

    parameters = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y)
    parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2







