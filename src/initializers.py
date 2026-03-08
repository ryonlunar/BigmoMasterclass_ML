import random
import math


# Zero initialization
def initialize_zero(num_of_neurons: int) -> list[float]:
    # isi semua weight pake nol
    return [0 for _ in range(num_of_neurons + 1)] # inisialisasi weight untuk semua neuron dan bias

# Random dengan distribusi uniform
def initialize_uniform(num_of_neurons: int, lower_bound: float, upper_bound: float, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(lower_bound, upper_bound) for _ in range(num_of_neurons + 1)]

# Random dengan distribusi normal
def initialize_random(num_of_neurons: int, mean: float, variance: float, seed: int | None = None) -> list[float]:
    rng = random.Random(seed)
    std_dev = math.sqrt(variance) # diambil akarnya karena random.normalvariate pakai standar deviasi, bukan varians
    return [rng.normalvariate(mean, std_dev) for _ in range(num_of_neurons + 1)]