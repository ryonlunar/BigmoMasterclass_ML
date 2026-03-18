import numpy as np
from .layer import Layer


class RMSNorm(Layer):
    def __init__(self, units, eps=1e-8, **kwargs):
        kwargs.pop('activation', None)
        super().__init__(units, **kwargs)
        self.eps = eps
        self.gamma = None
        self.d_gamma = None
        # Atribut untuk adam optimizer (di-inisialisasi di method build)
        self.m_gamma = None
        self.v_gamma = None
        self.t = None

    def build(self, input_dim):
        self.input_dim = input_dim
        self.W = np.zeros((input_dim, self.units))
        self.b = np.zeros((1, self.units))
        self.gamma = np.ones((1, self.units))
        self.dW = None
        self.db = None
        self.d_gamma = None
        # Initialize atribut untuk Adam
        self.m_gamma = np.zeros((1, self.units))
        self.v_gamma = np.zeros((1, self.units))
        self.t = 0

    def forward(self, input_data):
        self.input = input_data
        rms = np.sqrt(np.mean(input_data ** 2, axis=-1, keepdims=True) + self.eps)
        self.rms = rms
        self.x_norm = input_data / rms
        self.output = self.gamma * self.x_norm
        return self.output

    def backward(self, output_error, learning_rate):
        n = self.input.shape[-1]

        self.d_gamma = np.sum(output_error * self.x_norm, axis=0, keepdims=True)

        dx_norm = output_error * self.gamma
        drms = np.sum(dx_norm * self.input, axis=-1, keepdims=True) * (-1 / self.rms ** 2)

        input_error = dx_norm / self.rms + drms * (self.input / (n * self.rms))

        if self.use_adam:
            self.t += 1
            self.m_gamma = self.beta1 * self.m_gamma + (1 - self.beta1) * self.d_gamma
            self.v_gamma = self.beta2 * self.v_gamma + (1 - self.beta2) * (self.d_gamma ** 2)
            m_gamma_hat = self.m_gamma / (1 - self.beta1 ** self.t)
            v_gamma_hat = self.v_gamma / (1 - self.beta2 ** self.t)
            self.gamma -= learning_rate * m_gamma_hat / (np.sqrt(v_gamma_hat) + self.epsilon)
        else:
            self.gamma -= learning_rate * self.d_gamma

        return input_error
