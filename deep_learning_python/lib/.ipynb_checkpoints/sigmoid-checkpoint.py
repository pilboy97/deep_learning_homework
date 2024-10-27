import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None  # 순전파 결과를 저장

    def forward(self, x):
        """
        Sigmoid 활성화 함수의 순전파
        
        Parameters:
        - x: 입력 데이터 (numpy array)
        
        Returns:
        - sigmoid(x)
        """
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        """
        Sigmoid 활성화 함수의 역전파
        
        Parameters:
        - dout: 순전파의 출력에 대한 손실 기울기 (numpy array)
        
        Returns:
        - dx: 입력에 대한 기울기
        """
        dx = dout * self.out * (1 - self.out)
        return dx