import numpy as np
import math

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

def initialize_xavier_uniform(seed: int | None = None):
    def initer(shape: tuple):
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        rng = np.random.default_rng(seed)
        return rng.uniform(-limit, limit, size=shape)
    return initer

def initialize_he(seed: int | None = None):
    def initer(shape: tuple):
        fan_in = shape[0]
        std_dev = np.sqrt(2.0 / fan_in)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, std_dev, size=shape)
    return initer