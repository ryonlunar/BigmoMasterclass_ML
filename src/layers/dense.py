import numpy as np
from .layer import Layer
from activations import Activations

class Dense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(units, **kwargs)
        self.activation_func = None
        self.d_activation_func = None
        # ambil fungsi aktivasi dan turunannya
        if activation:
            self.activation_func, self.d_activation_func = Activations.get(activation)

    def forward(self, input_data):
        self.input = input_data
        # Z = X.W + b
        self.z = np.dot(self.input, self.W) + self.b
        # terapkan fungsi aktivasi
        if self.activation_func:
            self.output = self.activation_func(self.z)
        else:
            self.output = self.z
            
        return self.output
    
    def backward(self, output_error, learning_rate):
        # hitung jumlah data dalam satu batch
        batch_size = self.input.shape[0]
        
        # hitung Delta (dL/dZ) : dL/dA * dA/dZ
        if self.d_activation_func:
            delta = output_error * self.d_activation_func(self.z)
        else:
            delta = output_error

        # hitung gradien untuk W dan b
        # dL/dW = X.T * delta
        self.dW = np.dot(self.input.T, delta)
        if self.l1 > 0:
            self.dW += self.l1 * np.sign(self.W)
        if self.l2 > 0:
            self.dW += self.l2 * self.W

        # dL/db = delta
        self.db = np.sum(delta, axis=0, keepdims=True)

        # hitung gradien untuk input
        # dL/dX = delta * W.T
        input_error = np.dot(delta, self.W.T)

        # update weights dan bias
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

        return input_error