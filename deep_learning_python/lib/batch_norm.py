import numpy as np

class BatchNorm2d:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        self.gamma = np.ones((1, num_features, 1, 1))
        self.beta = np.zeros((1, num_features, 1, 1))
        
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
        self.x_hat = None
        self.mean = None
        self.var = None

    def forward(self, x, training=True):
        """
        Forward pass for batch normalization
        
        Parameters:
        - x: Input array (batch_size, num_features, height, width)
        - training: Boolean, whether the layer is in training mode
        
        Returns:
        - Output: Batch normalized output
        """
        if training:
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            self.x_hat = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            self.mean = batch_mean
            self.var = batch_var
        else:
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        batch_size, _, height, width = dout.shape
        
        dgamma = np.sum(dout * self.x_hat, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        
        dx_hat = dout * self.gamma
        
        dvar = np.sum(dx_hat * (self.x_hat * -0.5) / (self.var + self.epsilon), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.epsilon), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2 * (self.x_hat), axis=(0, 2, 3), keepdims=True)
        
        dx = dx_hat / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.x_hat) / (batch_size * height * width) + dmean / (batch_size * height * width)
        
        return dx, dgamma, dbeta
