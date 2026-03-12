from .layer import Layer
class Dense(Layer):
    def __init__(self, units):
        super().__init__(units)
        
    def forward(self, input_data):
        pass
    
    def backward(self, output_error, learning_rate):
        pass