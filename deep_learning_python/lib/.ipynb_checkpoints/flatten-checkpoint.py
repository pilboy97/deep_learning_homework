class Flatten:
    def forward(self, X):
        self.input_shape = X.shape  # 입력 데이터의 원래 형상 저장
        return X.reshape(X.shape[0], -1)  # 배치 크기를 유지한 채 평탄화

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)  # 역전파 시 원래 차원으로 복구
