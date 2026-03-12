import random
import math

def initialize_zero():
    def initer(num_of_neurons: int):
        return [0 for _ in range(num_of_neurons + 1)] 
    return initer

def initialize_uniform(lower_bound: float, upper_bound: float, seed: int | None = None):
    def initer(num_of_neurons : int):        
        rng = random.Random(seed)
        return [rng.uniform(lower_bound, upper_bound) for _ in range(num_of_neurons + 1)]
    return initer

def initialize_random(mean: float, variance: float, seed: int | None = None):
    def initer(num_of_neurons : int):
        rng = random.Random(seed)
        std_dev = math.sqrt(variance)
        return [rng.normalvariate(mean, std_dev) for _ in range(num_of_neurons + 1)]
    return initer