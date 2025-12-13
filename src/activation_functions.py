import numpy as np

class ReLU():
    """
    Defines the ReLU activation function for the neural network
    """

    def __init__(self):
        self.Z = None   # ReLU pre-activation inputs
    
    def forward(self, Z):
        """
        Calculate a forward pass for this layer

        args:
            Z - ReLU pre-activaiton inputs
        
        returns:
            A - activation output
        """

        self.Z = Z
        A = np.maximum(Z, 0)

        return A
    
    def backward(self, dA):
        """
        Calculate a backwards pass for this layer

        args:
            dA - gradient of loss w.r.t layer's output and shape
        
        returns:
            dX - gradient of loss w.r.t layer's input
        """

        dX = np.where(self.Z > 0, dA, 0)

        return dX

class SoftmaxCrossEntropy():
    """
    Defines the Softmax activation function for the neural network
    """

    def __init__(self):
        self.A = None   # activation output
        self.Y = None   # True labels
    
    def forward(self, Z, Y):
        """
        Calculate a forward pass for this layer

        args:
            Z - ReLU pre-activaiton inputs
        
        returns:
            L - Loss value
        """

        exp_Z = np.exp(Z-np.max(Z, axis=1, keepdims=True))
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        A = np.clip(A, 1e-12, 1.0) # clip for numerical stability

        self.A = A
        self.Y = Y

        L = -np.mean(np.sum(Y*np.log(A), axis=1))

        return L
    
    def backward(self):
        """
        Calculate a backwards pass for this layer
        
        returns:
            dZ - gradient of loss w.r.t layer's input
        """

        dZ = (self.A - self.Y) / self.Y.shape[0]
        return dZ