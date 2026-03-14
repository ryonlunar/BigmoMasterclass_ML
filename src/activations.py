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
    @staticmethod
    def softmax(inputs: np.ndarray) -> np.ndarray:
        exp_inputs = np.exp(inputs)
        sum_exp_inputs = np.sum(exp_inputs)
        return exp_inputs / sum_exp_inputs # bagi exp(input) dengan jumlah exp(input)
    @staticmethod
    def softmax_derivative(inputs: np.ndarray) -> np.ndarray:
        softmax_output = Activations.softmax(inputs)
        # ∂s_i/∂x_j adalah turunan dari softmax_i terhadap input_j
        # = s_i * (δ_ij − s_j) = (s_i * δ_ij) - (s_i * s_j)
        # dengan δ_ij = 1 jika i == j dan δ_ij = 0 ketika i != j
        # maka, hal tersebut bisa direpresentasikan dengan matriks diagonal (pemanggilan method np.diagflat() di bawah)
        # karena matriks diagonal memiliki diagonal 1 dan elemen sisanya 0, berarti matriks np.diagflat(softmax_output) memiliki elemen diagonal s_i dan sisanya 0
        # hal tersebut merepresentasikan bagian (s_i * δ_ij) dari rumus
        # lalu (s_i * s_j) direpresentasikan oleh hasil perkalian (outer product) dari matriks softmax dengan dirinya sendiri
        return np.diagflat(softmax_output) - np.outer(softmax_output, softmax_output)