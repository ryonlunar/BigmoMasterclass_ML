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

        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        return input_error
