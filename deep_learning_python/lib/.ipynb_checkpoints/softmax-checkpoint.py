import numpy as np

class Softmax:
    def __init__(self):
        self.y_pred = None
    
    def forward(self, X, y_true):
        # X: (batch_size, num_classes, height, width)
        # y_true: (batch_size, 1, height, width) - num_channel 은 항상 1로 가정
        
        # Softmax 계산
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.y_pred = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        
        # Flatten the dimensions for easy indexing
        n_samples, num_classes, height, width = X.shape
        self.y_pred = self.y_pred.transpose(0, 2, 3, 1).reshape(-1, num_classes)  # Shape: (batch_size * height * width, num_classes)
        y_true = y_true.reshape(n_samples, -1)  # Shape: (batch_size, height * width)
        
        # Convert y_true to flat indices and ensure they are integers
        flat_y_true = y_true.flatten().astype(int)

        if flat_y_true.shape[0] != self.y_pred.shape[0]:
            raise ValueError(f"Shape mismatch: y_pred has {self.y_pred.shape[0]} elements, but y_true has {flat_y_true.shape[0]} elements.")

        # Calculate loss using cross-entropy
        log_likelihood = -np.log(self.y_pred[np.arange(self.y_pred.shape[0]), flat_y_true] + 1e-8)
        loss = np.sum(log_likelihood) / self.y_pred.shape[0]
        
        return loss
        
    def backward(self, y_true):
        # y_true: (batch_size, 1, height, width)
        height, width = self.y_pred.shape[2], self.y_pred.shape[3]
        n_samples, num_classes = self.y_pred.shape[0] // (height * width), self.y_pred.shape[1]
    
        # Flatten the y_pred and y_true for gradient calculation
        grad = self.y_pred.copy()  # Shape: (batch_size * height * width, num_classes)
        y_true_flat = y_true.reshape(-1).astype(int)  # Flattened version

        if grad.shape[0] != y_true_flat.shape[0]:
            raise ValueError(f"Shape mismatch in backward: y_pred has {grad.shape[0]} elements, but y_true has {y_true_flat.shape[0]} elements.")
        
        # Adjust grad values for the correct class labels
        grad[np.arange(grad.shape[0]), y_true_flat] -= 1
        
        # Reshape back to the original batch format (batch_size, num_classes, height, width)
        grad = grad.reshape(n_samples, height, width, num_classes).transpose(0, 3, 1, 2)
        
        return grad / (n_samples * height * width)
