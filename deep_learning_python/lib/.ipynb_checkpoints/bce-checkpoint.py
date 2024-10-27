import cupy as np

class BCE:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self.y_true = None
        self.y_pred = None
    
    def forward(self, y_true, y_pred):
        """
        Forward pass to compute BCE loss.
        
        Parameters:
        - y_true: numpy array of true labels (0 or 1)
        - y_pred: numpy array of predicted probabilities (between 0 and 1)
        
        Returns:
        - loss: scalar value of the BCE loss
        """
        # Store true labels and predictions for backward pass
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute BCE loss
        loss = -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        return loss
    
    def backward(self):
        """
        Backward pass to compute gradients with respect to y_pred.
        
        Returns:
        - dL/dy_pred: numpy array of gradients of the loss w.r.t. y_pred
        """
        # Ensure y_true and y_pred are set
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Must call forward() before backward().")
        
        # Compute gradient: dL/dy_pred
        grad = (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred))
        grad /= self.y_true.shape[0]  # Normalize by the batch size
        return grad