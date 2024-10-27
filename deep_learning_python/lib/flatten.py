class Flatten:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)
