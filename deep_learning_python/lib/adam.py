import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate  # 학습률
        self.beta1 = beta1       # 1차 모멘텀을 위한 계수
        self.beta2 = beta2       # 2차 모멘텀을 위한 계수
        self.epsilon = epsilon   # 수치적 안정성을 위한 작은 값
        self.m = {}              # 1차 모멘텀 추적
        self.v = {}              # 2차 모멘텀 추적
        self.t = 1               # 타임스텝 (1부터 시작하도록 변경)

    def update(self, layer):
        if not hasattr(layer, 'W'):
            return  # 가중치가 없는 레이어는 업데이트하지 않음

        self.t += 1

        if layer not in self.m:
            self.m[layer] = {}
            self.v[layer] = {}
            self.m[layer]['W'] = np.zeros_like(layer.W)
            self.m[layer]['b'] = np.zeros_like(layer.b)
            self.v[layer]['W'] = np.zeros_like(layer.W)
            self.v[layer]['b'] = np.zeros_like(layer.b)

        dW = layer.dW
        db = layer.db

        self.m[layer]['W'] = self.beta1 * self.m[layer]['W'] + (1 - self.beta1) * dW
        self.m[layer]['b'] = self.beta1 * self.m[layer]['b'] + (1 - self.beta1) * db
        self.v[layer]['W'] = self.beta2 * self.v[layer]['W'] + (1 - self.beta2) * (dW ** 2)
        self.v[layer]['b'] = self.beta2 * self.v[layer]['b'] + (1 - self.beta2) * (db ** 2)

        m_hat_W = self.m[layer]['W'] / (1 - self.beta1 ** self.t + self.epsilon)
        m_hat_b = self.m[layer]['b'] / (1 - self.beta1 ** self.t + self.epsilon)
        v_hat_W = self.v[layer]['W'] / (1 - self.beta2 ** self.t + self.epsilon)
        v_hat_b = self.v[layer]['b'] / (1 - self.beta2 ** self.t + self.epsilon)

        layer.W -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    def update_batchnorm_params(self, layer, dgamma, dbeta):
        if layer not in self.m:
            self.m[layer] = {}
            self.v[layer] = {}
            self.m[layer]['gamma'] = np.zeros_like(layer.gamma)
            self.m[layer]['beta'] = np.zeros_like(layer.beta)
            self.v[layer]['gamma'] = np.zeros_like(layer.gamma)
            self.v[layer]['beta'] = np.zeros_like(layer.beta)

        self.m[layer]['gamma'] = self.beta1 * self.m[layer]['gamma'] + (1 - self.beta1) * dgamma
        self.m[layer]['beta'] = self.beta1 * self.m[layer]['beta'] + (1 - self.beta1) * dbeta
        self.v[layer]['gamma'] = self.beta2 * self.v[layer]['gamma'] + (1 - self.beta2) * (dgamma ** 2)
        self.v[layer]['beta'] = self.beta2 * self.v[layer]['beta'] + (1 - self.beta2) * (dbeta ** 2)

        m_hat_gamma = self.m[layer]['gamma'] / (1 - self.beta1 ** self.t + self.epsilon)
        m_hat_beta = self.m[layer]['beta'] / (1 - self.beta1 ** self.t + self.epsilon)
        v_hat_gamma = self.v[layer]['gamma'] / (1 - self.beta2 ** self.t + self.epsilon)
        v_hat_beta = self.v[layer]['beta'] / (1 - self.beta2 ** self.t + self.epsilon)

        layer.gamma -= self.lr * m_hat_gamma / (np.sqrt(v_hat_gamma) + self.epsilon)
        layer.beta -= self.lr * m_hat_beta / (np.sqrt(v_hat_beta) + self.epsilon)
