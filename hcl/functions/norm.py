import numpy as np


def normalize(X: np.ndarray, tol=1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + tol)
