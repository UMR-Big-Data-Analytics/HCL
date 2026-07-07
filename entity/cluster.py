from dataclasses import dataclass

import numpy as np


@dataclass
class Cluster:
    id: int
    N: int
    S: np.ndarray
    R: float

    @classmethod
    def singleton(cls, idx: int, x: np.ndarray):
        return cls(id=idx, N=1, S=x.copy(), R=float(np.linalg.norm(x)))

    @staticmethod
    def merge_clusters(A: Cluster, B: Cluster, new_id: int) -> Cluster:
        S = A.S + B.S
        return Cluster(id=new_id, N=A.N + B.N, S=S, R=float(np.linalg.norm(S)))
