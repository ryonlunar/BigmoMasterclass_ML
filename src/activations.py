import numpy as np
from typing import Tuple, Callable


class Activations:
    @staticmethod
    def get(name: str) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        name = name.lower()
        if name == 'linear':  return Activations.linear,  Activations.d_linear
        if name == 'relu':    return Activations.relu,    Activations.d_relu
        if name == 'sigmoid': return Activations.sigmoid, Activations.d_sigmoid
        if name == 'tanh':    return Activations.tanh,    Activations.d_tanh
        if name == 'softmax': return Activations.softmax, Activations.d_softmax
        raise ValueError(f"Activation '{name}' not supported")

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        return x

    @staticmethod
    def d_linear(x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def d_relu(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def d_sigmoid(x: np.ndarray) -> np.ndarray:
        s = Activations.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def d_tanh(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def d_softmax(x: np.ndarray) -> np.ndarray:
        # ∂s_i/∂x_j adalah turunan dari softmax_i terhadap input_j
        # = s_i * (δ_ij − s_j) = (s_i * δ_ij) - (s_i * s_j)
        # dengan δ_ij = 1 jika i == j dan δ_ij = 0 ketika i != j
        # maka, hal tersebut bisa direpresentasikan dengan matriks diagonal (pemanggilan method np.diagflat() di bawah)
        # karena matriks diagonal memiliki diagonal 1 dan elemen sisanya 0, berarti matriks np.diagflat(softmax_output) memiliki elemen diagonal s_i dan sisanya 0
        # hal tersebut merepresentasikan bagian (s_i * δ_ij) dari rumus
        # lalu (s_i * s_j) direpresentasikan oleh hasil perkalian (outer product) dari matriks softmax dengan dirinya sendiri
        s = Activations.softmax(x)
        if x.ndim == 1:
            return np.diagflat(s) - np.outer(s, s)
        batch_size, n = s.shape
        J = np.zeros((batch_size, n, n))
        for i in range(batch_size):
            si = s[i]
            J[i] = np.diagflat(si) - np.outer(si, si)
        return J
