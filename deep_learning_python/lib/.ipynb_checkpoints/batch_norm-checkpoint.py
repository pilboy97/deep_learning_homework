import cupy as np

class BatchNorm2d:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        """
        Batch Normalization for 2D inputs (like images)
        
        Parameters:
        - num_features: Number of channels in the input
        - epsilon: Small value to prevent division by zero
        - momentum: Momentum for running mean and variance
        """
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # 학습 가능한 파라미터 (초기화)
        self.gamma = np.ones((1, num_features, 1, 1))  # 스케일링 파라미터
        self.beta = np.zeros((1, num_features, 1, 1))  # 시프트 파라미터
        
        # 러닝 스탯 (추정된 평균과 분산)
        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))
        
        # 역전파에 사용될 값들
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
            # 배치의 평균과 분산 계산
            batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
            batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)
            
            # 러닝 평균과 분산 업데이트 (지수 이동 평균)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # 정규화
            self.x_hat = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            self.mean = batch_mean
            self.var = batch_var
        else:
            # 러닝 평균과 분산 사용 (평가 모드)
            self.x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # 스케일과 시프트 적용
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        """
        Backward pass for batch normalization
        
        Parameters:
        - dout: Upstream gradient
        
        Returns:
        - dx: Gradient with respect to input
        - dgamma: Gradient with respect to gamma
        - dbeta: Gradient with respect to beta
        """
        batch_size, _, height, width = dout.shape
        
        # 파라미터 기울기
        dgamma = np.sum(dout * self.x_hat, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        
        # 정규화된 입력 기울기
        dx_hat = dout * self.gamma
        
        # 입력 기울기
        dvar = np.sum(dx_hat * (self.x_hat * -0.5) / (self.var + self.epsilon), axis=(0, 2, 3), keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.epsilon), axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2 * (self.x_hat), axis=(0, 2, 3), keepdims=True)
        
        dx = dx_hat / np.sqrt(self.var + self.epsilon) + dvar * 2 * (self.x_hat) / (batch_size * height * width) + dmean / (batch_size * height * width)
        
        return dx, dgamma, dbeta
