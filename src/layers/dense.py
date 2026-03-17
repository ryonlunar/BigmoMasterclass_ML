import numpy as np
from .layer import Layer
from activations import Activations


class Dense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(units, **kwargs)
        self.activation_func = None
        self.d_activation_func = None
        if activation:
            self.activation_func, self.d_activation_func = Activations.get(activation)
        # atribut untuk Adam optimizer, nanti di-initialize pakai method build()
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None
        self.t = 0

    def build(self, input_dim):
        self.input_dim = input_dim
        self.W = self.weight_initializer((input_dim, self.units))
        self.b = np.zeros((1, self.units))
        # Initialize atribut untuk Adam
        self.m_W = np.zeros((input_dim, self.units))
        self.v_W = np.zeros((input_dim, self.units))
        self.m_b = np.zeros((1, self.units))
        self.v_b = np.zeros((1, self.units))
        self.t = 0

    def forward(self, input_data):
        self.input = input_data
        self.z = np.dot(self.input, self.W) + self.b
        if self.activation_func:
            self.output = self.activation_func(self.z)
        else:
            self.output = self.z
        return self.output

    def backward(self, output_error, learning_rate):
        # hitung Delta (dL/dZ) : dL/dA * dA/dZ
        if self.d_activation_func:
            d_act = self.d_activation_func(self.z)
            if d_act.ndim == 3:
                delta = np.einsum('bi,bij->bj', output_error, d_act)
            else:
                delta = output_error * d_act
        else:
            delta = output_error

        self.dW = np.dot(self.input.T, delta)
        if self.l1 > 0:
            self.dW += self.l1 * np.sign(self.W)
        if self.l2 > 0:
            self.dW += self.l2 * self.W

        self.db = np.sum(delta, axis=0, keepdims=True)

        input_error = np.dot(delta, self.W.T)

        if self.use_adam:
            self.t += 1
            self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * self.dW
            self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (self.dW ** 2)
            m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
            self.W -= learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)

            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * self.db
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (self.db ** 2)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
            self.b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db

        return input_error
