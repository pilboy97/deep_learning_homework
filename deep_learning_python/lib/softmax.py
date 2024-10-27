import numpy as np

class Softmax:
    def __init__(self):
        self.y_pred = None
    
    def forward(self, X, y_true):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.y_pred = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        
        n_samples, num_classes, height, width = X.shape
        self.y_pred = self.y_pred.transpose(0, 2, 3, 1).reshape(-1, num_classes)
        y_true = y_true.reshape(n_samples, -1)
        
        flat_y_true = y_true.flatten().astype(int)

        if flat_y_true.shape[0] != self.y_pred.shape[0]:
            raise ValueError(f"Shape mismatch: y_pred has {self.y_pred.shape[0]} elements, but y_true has {flat_y_true.shape[0]} elements.")

        log_likelihood = -np.log(self.y_pred[np.arange(self.y_pred.shape[0]), flat_y_true] + 1e-8)
        loss = np.sum(log_likelihood) / self.y_pred.shape[0]
        
        return loss
        
    def backward(self, y_true):
        height, width = y_true.shape[2], y_true.shape[3]
        n_samples, num_classes = self.y_pred.shape[0] // (height * width), self.y_pred.shape[1]
    
        grad = self.y_pred.copy()
        y_true_flat = y_true.reshape(-1).astype(int)

        if grad.shape[0] != y_true_flat.shape[0]:
            raise ValueError(f"Shape mismatch in backward: y_pred has {grad.shape[0]} elements, but y_true has {y_true_flat.shape[0]} elements.")
        
        grad[np.arange(grad.shape[0]), y_true_flat] -= 1
        
        grad = grad.reshape(n_samples, height, width, num_classes).transpose(0, 3, 1, 2)
        
        return grad / (n_samples * height * width)
