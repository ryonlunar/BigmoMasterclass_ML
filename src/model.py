import numpy as np
from .layers.layer import Layer

class FFNN:
    def __init__(self):
        self.layer_list : list[Layer] = []
    
    def add(self, layer : Layer):
        self.layer_list.append(layer)
        
    def compile(self):
        if len(self.layer_list) == 0:
            raise ValueError("Model tidak memiliki layer")

        input_size = self.layer_list[0].input_dim

        if input_size is None:
            raise ValueError("Layer pertama harus memiliki input_dim")

        for layer in self.layer_list:
            layer.build(input_size)
            input_size = layer.units
            
    def predict(self, input_data):
        pass
    
    def fit(self, train, test, epoch = 10):
        for _ in range(epoch):
            output = []
            for layer in self.layer_list:
                output.append(layer.forward(train if len(output) == 0 else output[-1]))
            
            for layer in self.layer_list[:-1] :
                pass
                