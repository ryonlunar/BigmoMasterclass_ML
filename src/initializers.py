import numpy as np

def initialize_zero():
    def initer(shape: tuple):
        return np.zeros(shape)
    return initer

def initialize_uniform(lower_bound: float, upper_bound: float, seed: int | None = None):
    def initer(shape: tuple):        
        rng = np.random.default_rng(seed)
        return rng.uniform(lower_bound, upper_bound, size=shape)
    return initer

def initialize_random(mean: float, variance: float, seed: int | None = None):
    def initer(shape: tuple):
        rng = np.random.default_rng(seed)
        std_dev = np.sqrt(variance)
        return rng.normal(mean, std_dev, size=shape)
    return initer