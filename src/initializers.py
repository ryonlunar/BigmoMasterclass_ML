import numpy as np


class initialize_zero:
    def __call__(self, shape: tuple):
        return np.zeros(shape)


class initialize_uniform:
    def __init__(self, lower_bound: float, upper_bound: float, seed: int | None = None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.seed = seed

    def __call__(self, shape: tuple):
        rng = np.random.default_rng(self.seed)
        return rng.uniform(self.lower_bound, self.upper_bound, size=shape)


class initialize_random:
    def __init__(self, mean: float, variance: float, seed: int | None = None):
        self.mean = mean
        self.variance = variance
        self.seed = seed

    def __call__(self, shape: tuple):
        rng = np.random.default_rng(self.seed)
        std_dev = np.sqrt(self.variance)
        return rng.normal(self.mean, std_dev, size=shape)


class initialize_xavier_uniform:
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def __call__(self, shape: tuple):
        fan_in, fan_out = shape[0], shape[1]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        rng = np.random.default_rng(self.seed)
        return rng.uniform(-limit, limit, size=shape)


class initialize_he:
    def __init__(self, seed: int | None = None):
        self.seed = seed

    def __call__(self, shape: tuple):
        fan_in = shape[0]
        std_dev = np.sqrt(2.0 / fan_in)
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, std_dev, size=shape)
