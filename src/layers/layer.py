import numpy as np
from abc import ABC, abstractmethod
from initializers import initialize_zero

class Layer(ABC):
    def __init__ (self, units, input_dim = None, activation = None, weight_initializer = initialize_zero()):
        self.input_dim = input_dim
        self.units = units
        self.W = None
        self.b = None
        self.activation = activation
        self.weight_initializer = weight_initializer
        
    def build(self, input_dim):
        self.input_dim = input_dim
        self.W = self.weight_initializer((input_dim +1 )*self.units)
        self.b = np.zeros(self.units)
    
    @abstractmethod
    def forward(self, input_data):
        pass
    
    @abstractmethod
    def backward(self, output_error, learning_rate):
        pass
