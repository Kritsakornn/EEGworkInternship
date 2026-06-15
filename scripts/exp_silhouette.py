"""THROWAWAY experiment (cluster-validity check before the professor meeting; safe to delete).

Answers: do the 33-dim RSP features actually contain 3 well-separated clusters,
or is the K=3 partition just an algorithmic cut through one blurry cloud?

Silhouette score (euclidean, on a subsample because full 84k^2 is too big):
  ~1.0 = tight, well-separated clusters; ~0 = clusters overlap / no real structure.
  Part A: silhouette vs K=2..5 (is K=3 a peak, or does quality just fall?).
  Part B: K=3 silhouette across 5 seeds (does the random seed change quality too?).

Run:  PYTHONPATH=src python3 scripts/exp_silhouette.py
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

import config
from macrostates import complete_subjects, build_matrix, kmedoids

SUB_N = 5000  # subsample size for tractable silhouette


def fit_labels(D, k, rng, n_restarts=config.N_RESTARTS):
    """K-medoids on the subsample distance matrix; keep lowest eq.4 cost -> labels."""
    best = None
    for _ in range(n_restarts):
        medoids = kmedoids(D, k, rng)
        dist = D[:, medoids]
        cost = dist.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, dist.argmin(axis=1))
    return best[1]


subs = complete_subjects()
X, _ = build_matrix(subs)
S = X[np.random.default_rng(0).choice(len(X), SUB_N, replace=False)]
D = cdist(S, S, metric="sqeuclidean")  # clustering uses sqeuclidean (paper eq.4)
print(f"X={X.shape}; silhouette on a {SUB_N}-point subsample (euclidean)\n")

print("=== Part A: silhouette vs K (seed 0) ===")
for k in (2, 3, 4, 5):
    lab = fit_labels(D, k, np.random.default_rng(0))
    print(f"  K={k}: silhouette={silhouette_score(S, lab, metric='euclidean'):.3f}")

print("\n=== Part B: K=3 silhouette across seeds 0-4 ===")
for seed in range(5):
    lab = fit_labels(D, 3, np.random.default_rng(seed))
    print(f"  seed {seed}: silhouette={silhouette_score(S, lab, metric='euclidean'):.3f}")
