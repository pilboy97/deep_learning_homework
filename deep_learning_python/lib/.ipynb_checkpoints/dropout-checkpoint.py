import cupy as np

class Dropout:
    def __init__(self, drop_prob=0.5):
        """
        드롭아웃 레이어 초기화
        :param drop_prob: 드롭아웃 확률 (0.5는 절반의 뉴런을 드롭)
        """
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, X, is_training=True):
        """
        순전파 (Forward Pass)
        :param X: 입력 데이터
        :param is_training: 학습 중인 경우 True, 테스트 중인 경우 False
        :return: 드롭아웃 적용된 출력
        """
        if is_training:
            # 무작위로 뉴런을 비활성화하는 마스크 생성
            self.mask = np.random.rand(*X.shape) > self.drop_prob
            # 마스크 적용 후 스케일링 (드롭아웃 비율에 따라 출력 값을 보정)
            return X * self.mask / (1 - self.drop_prob)
        else:
            # 테스트 시에는 드롭아웃 없이 스케일링된 출력을 사용
            return X

    def backward(self, d_out):
        """
        역전파 (Backward Pass)
        :param d_out: 출력 기울기
        :return: 입력 기울기
        """
        # 학습 시 적용된 마스크를 사용하여 역전파 기울기를 계산
        return d_out * self.mask / (1 - self.drop_prob)
