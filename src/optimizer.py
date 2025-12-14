from typing import List
from layer import Layer

class SGD():
    """
    Update network parameters via stochastic gradient descent
    """

    def __init__(self, parameters: List[Layer], learning_rate):
        self.layers = parameters
        self.learning_rate = learning_rate

    def step(self):
        """
        Apply gradient descent
        """

        for layer in self.layers:
            layer.W = layer.W - self.learning_rate*layer.dW
            layer.b = layer.b - self.learning_rate*layer.db

    def zero_grad(self):
        """
        Reset gradients
        """

        for layer in self.layers:
            layer.dW = None
            layer.db = None