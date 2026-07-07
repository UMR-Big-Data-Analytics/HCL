import numpy as np


def normalize(X: np.ndarray, tol=1e-12) -> np.ndarray:
    # numpy norm does not work with numba with axis arg
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + tol)
