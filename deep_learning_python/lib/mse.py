import numpy as np

class MSE:
    def forward(self, y_true, y_pred):
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true):
        n = y_true.shape[0]  # 데이터 포인트 수
        loss = 2 * (self.y_pred - y_true) / n

        return loss