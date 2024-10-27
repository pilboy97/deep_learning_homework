import numpy as np

class BCE:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.y_true = None
        self.y_pred = None
    
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        loss = -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        return loss
    
    def backward(self):
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call forward() before backward().")
        
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        grad /= self.y_true.shape[0]
        return grad