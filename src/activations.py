import math
from typing import Tuple, Callable
import numpy as np
class Activations:
    @staticmethod
    def get(name: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        if name == 'sigmoid': return Activations.sigmoid, Activations.d_sigmoid
        if name == 'tanh': return Activations.tanh, Activations.d_tanh
        # TODO : add other activation func
        raise ValueError(f"Activation '{name}' not supported")
    
    @staticmethod
    def linear():
        pass

    @staticmethod
    def relu():
        pass

    @staticmethod
    def sigmoid(x: np.ndarray)-> np.ndarray:
        x = np.clip(x, -500, 500)
        ex = np.exp(-x)
        return 1/(1+ex)
    
    @staticmethod
    def d_sigmoid(x: np.ndarray)->np.ndarray:
        s = Activations.sigmoid(x)
        return s * (1-s)
    
    @staticmethod
    def tanh(x:np.ndarray)->np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def d_tanh(x:np.ndarray)->np.ndarray:
        a = Activations.tanh(x)
        return 1 - a**2

        
    # softmax
    def softmax(inputs: list[float]) -> list[float]:
        exp_inputs = [math.exp(xi) for xi in inputs]
        sum_exp_inputs = sum(exp_inputs)
        return [ei / sum_exp_inputs for ei in exp_inputs]