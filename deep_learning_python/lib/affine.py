import numpy as np
class Affine:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.b

    def backward(self, d_out, train=True):
        if train:
            self.dW = np.dot(self.X.T, d_out)
            self.db = np.sum(d_out, axis=0, keepdims=True)

        dX = np.dot(d_out, self.W.T)
        return dX
