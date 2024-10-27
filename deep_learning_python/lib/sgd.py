class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def update(self, layer):
        if hasattr(layer, 'W'):
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db
