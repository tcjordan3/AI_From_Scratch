import numpy as np

class Layer():
    """
    Defines a layer of the neural network architecture
    """

    def __init__(self, input_dim, output_dim):
        # He initialization
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim) # weights
        self.b = np.zeros(output_dim)   # biases

        self.X = None   # inputs
        self.dW = None  # loss gradient w.r.t weights
        self.db = None  # loss gradient w.r.t biases

    def forward(self, X):
        """
        Calculate a forward pass for this layer

        returns:
            Z - pre-activation output array
        """

        # cache X for use in backpropogation
        self.X = X

        Z = np.dot(X, self.W) + self.b

        return Z
    
    def backward(self, dZ):
        """
        Calculate a backwards pass for this layer

        args:
            dZ - gradient of loss w.r.t layer's output and shape

        return:
            dX - gradient of loss w.r.t layer's input
        """

        dX = np.dot(dZ, self.W.T)
        self.dW = np.dot(self.X.T, dZ)
        self.db = np.sum(dZ, axis=0)

        return dX