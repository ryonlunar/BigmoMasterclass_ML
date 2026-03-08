import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__ (self) :
        self.input = None
        self.output = None
    
    @abstractmethod
    def forward(self, input_data):
        pass
    
    @abstractmethod
    def backward(self, output_error, learning_rate):
        pass
