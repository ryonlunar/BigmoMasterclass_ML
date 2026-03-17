import numpy as np
from abc import ABC, abstractmethod
from initializers import initialize_zero

class Layer(ABC):
    def __init__ (self, units, input_dim = None, activation = None, weight_initializer = initialize_zero(), l1=0.0, l2=0.0,
                  use_adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.input_dim = input_dim
        self.units = units
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.l1 = l1
        self.l2 = l2
        self.use_adam = use_adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def build(self, input_dim):
        self.input_dim = input_dim
        self.W = self.weight_initializer((input_dim, self.units))
        self.b = np.zeros((1, self.units))
    
    @abstractmethod
    def forward(self, input_data):
        pass
    
    @abstractmethod
    def backward(self, output_error, learning_rate):
        pass
