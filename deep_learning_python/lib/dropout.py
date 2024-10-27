import numpy as np

class Dropout:
    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, X, is_training=True):
        if is_training:
            self.mask = np.random.rand(*X.shape) > self.drop_prob
            return X * self.mask / (1 - self.drop_prob)
        else:
            return X

    def backward(self, d_out):
        return d_out * self.mask / (1 - self.drop_prob)
