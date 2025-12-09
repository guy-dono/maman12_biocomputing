import numpy as np


def blend_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    For each gene i, the offspring value is uniformly sampled from:
    [min(p1[i], p2[i]) - α*d, max(p1[i], p2[i]) + α*d]
    where d = |p1[i] - p2[i]|
    """
    min_vals = np.minimum(parent1, parent2)
    max_vals = np.maximum(parent1, parent2)
    d = max_vals - min_vals

    low = min_vals - alpha * d
    high = max_vals + alpha * d

    return np.random.uniform(low, high)