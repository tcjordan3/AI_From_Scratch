from layer import Layer
from activation_functions import ReLU, SoftmaxCrossEntropy

class Network():
    """
    Defines architecture of custom neural network
    """

    def __init__(self):
        """
        Architecture:
            X → Dense → ReLU → Dense → ReLU → Dense → SoftmaxCrossEntropy(Y)
        """

        self.dense_1 = Layer(784, 128)
        self.relu_1 = ReLU()
        self.dense_2 = Layer(128, 64)
        self.relu_2 = ReLU()
        self.dense_3 = Layer(64, 10)
        self.softmax = SoftmaxCrossEntropy()

    def forward(self, X, Y=None):
        """
        Full forward pass thorugh the network.
        Compute Loss if labels provided
        
        args:
            X - network input
            Y - labels

        returns:
            L - Loss value if labels provided
            A - output probability array if no labels provided
        """

        Z_1 = self.dense_1.forward(X)
        A_1 = self.relu_1.forward(Z_1)
        Z_2 = self.dense_2.forward(A_1)
        A_2 = self.relu_2.forward(Z_2)
        Z_3 = self.dense_3.forward(A_2)

        if Y is not None:
            L = self.softmax.forward(Z_3, Y)
            return L
        else:
            A = self.softmax.forward(Z_3)
            return A

    def backward(self):
        """
        Full backward pass through the network

        returns:
            dX - gradient of loss w.r.t network input
        """

        if self.softmax.Y is None:
            raise ValueError("Cannot call backward without running forward with labels first")

        dZ_3 = self.softmax.backward()
        dA_2 = self.dense_3.backward(dZ_3)
        dZ_2 = self.relu_2.backward(dA_2)
        dA_1 = self.dense_2.backward(dZ_2)
        dZ_1 = self.relu_1.backward(dA_1)
        dX = self.dense_1.backward(dZ_1)

        return dX

    def get_parameters(self):
        """
        Return dense layer objects for optimizer usage
        """

        return [self.dense_1, self.dense_2, self.dense_3]