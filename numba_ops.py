import numpy as np
from numba import njit


@njit(fastmath=True)
def calculate_feature_distance(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean distance between feature arrays using Numba JIT compilation.
    """
    return np.sqrt(f1**2 + f2**2)


@njit(fastmath=True)
def compute_numba_metrics(f1: np.ndarray, f2: np.ndarray):
    """
    Computes mathematical summary statistics over feature arrays.
    """
    distances = calculate_feature_distance(f1, f2)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    return mean_dist, std_dist