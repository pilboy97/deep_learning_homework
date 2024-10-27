import cupy as np

class MSE:
    def forward(self, y_true, y_pred):
        """
        MSE 손실 계산 (순전파)
        :param y_true: 실제 값 (cupy array)
        :param y_pred: 예측 값 (cupy array)
        :return: 손실 값 (scalar)
        """
        self.y_pred = y_pred
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true):
        """
        MSE 손실에 대한 기울기 계산 (역전파)
        :return: 기울기 (cupy array, y_pred와 동일한 shape)
        """
        n = y_true.shape[0]  # 데이터 포인트 수
        loss = 2 * (self.y_pred - y_true) / n

        return loss