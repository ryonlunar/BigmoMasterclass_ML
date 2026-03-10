import random
import math
import numpy as np


# Zero initialization
def initialize_zero(num_of_neurons: int) -> np.ndarray:
    # isi semua weight pake nol
    return np.zeros(num_of_neurons + 1) # inisialisasi weight untuk semua neuron dan bias

# Random dengan distribusi uniform
def initialize_uniform(num_of_neurons: int, lower_bound: float, upper_bound: float, seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([rng.uniform(lower_bound, upper_bound) for _ in range(num_of_neurons + 1)])

# Random dengan distribusi normal
def initialize_random(num_of_neurons: int, mean: float, variance: float, seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    std_dev = math.sqrt(variance) # diambil akarnya karena random.normalvariate pakai standar deviasi, bukan varians
    return np.array([rng.normalvariate(mean, std_dev) for _ in range(num_of_neurons + 1)])