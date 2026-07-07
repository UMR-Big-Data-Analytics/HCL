import numpy as np


def vmf_score(N: int, R: float, gamma: float) -> float:
    return N * np.log(1.0 - (R / N) + gamma)


def vmf_linkage(N_A: int, R_A: float, N_B: int, R_B: float, N_AB: int, R_AB: float, gamma: float) -> float:
    score_ab = vmf_score(N_AB, R_AB, gamma)
    score_a = vmf_score(N_A, R_A, gamma)
    score_b = vmf_score(N_B, R_B, gamma)

    cost = score_ab - score_a - score_b

    # cost = cost + 1e-4
    # assert cost >= 0, f"GLR cannot be negative. Was {cost}"
    return cost


def spherical_ward_linkage(S_A: np.ndarray, S_B: np.ndarray):
    cost = np.linalg.norm(S_A) + np.linalg.norm(S_B) - np.linalg.norm(S_A + S_B)
    # assert cost >= 0, "Spherical ward cannot be negative"
    return cost
