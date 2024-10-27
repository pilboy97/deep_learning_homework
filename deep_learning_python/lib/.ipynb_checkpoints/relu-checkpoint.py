import cupy as np

class ReLU:
    def __init__(self):
        pass
    
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, d_out):
        dX = d_out.copy()
        dX[self.X <= 0] = 0
        return dX
