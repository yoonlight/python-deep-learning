import numpy as np
from src.common.layers import Affine, Sigmoid


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size) -> None:
        I, H, O = input_size, hidden_size, output_size

        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.params = []
        for layer in self.layers:
            self.params += layer.params
        pass

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
