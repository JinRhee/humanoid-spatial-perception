from __future__ import annotations

import numpy as np


def cosine_affinity(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    if g1.size == 0 or g2.size == 0:
        return np.zeros((g1.shape[0], g2.shape[0]), dtype=np.float32)
    n1 = g1 / (np.linalg.norm(g1, axis=1, keepdims=True) + 1e-9)
    n2 = g2 / (np.linalg.norm(g2, axis=1, keepdims=True) + 1e-9)
    return (n1 @ n2.T).astype(np.float32)


def sinkhorn_logspace(scores: np.ndarray, iters: int = 50) -> np.ndarray:
    m, n = scores.shape
    dust = np.zeros((m + 1, n + 1), dtype=np.float32)
    dust[:m, :n] = scores

    logp = dust.copy()
    for _ in range(iters):
        logp = logp - np.logaddexp.reduce(logp, axis=1, keepdims=True)
        logp = logp - np.logaddexp.reduce(logp, axis=0, keepdims=True)
    return np.exp(logp)


def mutual_matches_with_dustbin(assign: np.ndarray) -> list[tuple[int, int, float]]:
    m = assign.shape[0] - 1
    n = assign.shape[1] - 1
    if m <= 0 or n <= 0:
        return []

    row_best = np.argmax(assign[:m, :], axis=1)
    col_best = np.argmax(assign[:, :n], axis=0)

    out: list[tuple[int, int, float]] = []
    for i in range(m):
        j = int(row_best[i])
        if j == n:
            continue
        if int(col_best[j]) != i:
            continue
        if int(np.argmax(assign[:, j])) == m:
            continue
        out.append((i, j, float(assign[i, j])))
    return out
