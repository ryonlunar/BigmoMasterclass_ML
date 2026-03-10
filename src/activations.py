import math
import numpy as np


def linear():
    pass

def relu():
    pass
    
# softmax
def softmax(inputs: np.ndarray) -> np.ndarray:
    exp_inputs = np.exp(inputs)
    sum_exp_inputs = np.sum(exp_inputs)
    return exp_inputs / sum_exp_inputs # bagi exp(input) dengan jumlah exp(input)
    
def softmax_derivative(inputs: np.ndarray) -> np.ndarray:
    softmax_output = softmax(inputs)
    # ∂s_i/∂x_j adalah turunan dari softmax_i terhadap input_j
    # = s_i * (δ_ij − s_j) = (s_i * δ_ij) - (s_i * s_j)
    # dengan δ_ij = 1 jika i == j dan δ_ij = 0 ketika i != j
    # maka, hal tersebut bisa direpresentasikan dengan matriks diagonal (pemanggilan method np.diagflat() di bawah)
    # karena matriks diagonal memiliki diagonal 1 dan elemen sisanya 0, berarti matriks np.diagflat(softmax_output) memiliki elemen diagonal s_i dan sisanya 0
    # hal tersebut merepresentasikan bagian (s_i * δ_ij) dari rumus
    # lalu (s_i * s_j) direpresentasikan oleh hasil perkalian (outer product) dari matriks softmax dengan dirinya sendiri
    return np.diagflat(softmax_output) - np.outer(softmax_output, softmax_output)