import numpy as np

from hcl.functions import normalize


def random_subset(X: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    idx = np.random.choice(X.shape[0], n, replace=False)
    return X[idx], y[idx]


def prepare_data(x: np.ndarray, y: np.ndarray, n: int = 1000, seed=42) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] > n:
        x, y = random_subset(x, y, n=n, seed=seed)
    else:
        x, y = x, y
    x = normalize(x)
    return x, y
