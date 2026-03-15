import numpy as np
from typing import Tuple, Callable


class Losses:
    @staticmethod
    def get(name: str) -> Tuple[Callable[[np.ndarray, np.ndarray], float], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
        name = name.lower()
        if name == 'mse': return Losses.mse, Losses.d_mse
        if name == 'bce': return Losses.bce, Losses.d_bce
        if name == 'cce': return Losses.cce, Losses.d_cce
        raise ValueError(f"Loss '{name}' not supported")

    @staticmethod
    def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.power(y - y_pred, 2)))

    @staticmethod
    def d_mse(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -2 * (y - y_pred) / y.size

    @staticmethod
    def bce(y: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

    @staticmethod
    def d_bce(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y.shape[0]
        return (1 / n) * (y_pred - y) / (y_pred * (1 - y_pred))

    @staticmethod
    def cce(y: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(np.sum(y * np.log(y_pred), axis=1)))

    @staticmethod
    def d_cce(y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y.shape[0]
        return -(1 / n) * y / y_pred
