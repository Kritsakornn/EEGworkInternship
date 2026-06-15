"""THROWAWAY experiment (for understanding before the professor meeting; safe to delete).

Answers two questions WITHOUT modifying the pipeline:
  Q1. Is the current K-medoids (sqeuclidean, 10 restarts) stable across random seeds?
  Q2. How different is the paper-literal Euclidean distance from the current squared one?

Run:  PYTHONPATH=src python3 scripts/exp_metric_stability.py
"""
import numpy as np
from scipy.spatial.distance import cdist

import config
from macrostates import (complete_subjects, build_matrix, kmedoids,
                         map_states, compute_sri, _frontal_theta_idx, STATES)


def run_once(X, metric, seed, n_restarts=config.N_RESTARTS, fit_n=4000):
    """Replicates cluster_frames() but with a chosen metric/seed; returns labels."""
    rng = np.random.default_rng(seed)
    fit_n = min(fit_n, len(X))
    sample = X[rng.choice(len(X), fit_n, replace=False)]
    D = cdist(sample, sample, metric=metric)
    best = None
    for _ in range(n_restarts):
        medoids = sample[kmedoids(D, config.N_CLUSTERS, rng)]
        dist = cdist(X, medoids, metric=metric)
        cost = dist.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, dist.argmin(axis=1))
    return best[1]


def theta_per_state(X, labels, names, idx):
    return {names[c]: X[labels == c][:, idx].mean() for c in range(config.N_CLUSTERS)}


def report(X, groups, labels, tag):
    names = map_states(X, labels)
    frontal = _frontal_theta_idx()
    allch_theta = [config.CHANNELS_11.index(c) * 3 + 0 for c in config.CHANNELS_11]
    ft = theta_per_state(X, labels, names, frontal)
    tt = theta_per_state(X, labels, names, allch_theta)
    sri = compute_sri(labels, groups, names)[STATES].mean()
    print(f"{tag}")
    print(f"   SRI mix      Reading/Reasoning/Decision = "
          f"{sri['Reading']:.3f}/{sri['Reasoning']:.3f}/{sri['Decision']:.3f}")
    print(f"   frontal theta (USED for naming)          = "
          f"{ft['Reading']:.3f}/{ft['Reasoning']:.3f}/{ft['Decision']:.3f}")
    print(f"   11-ch theta  (FREE check, not for naming) = "
          f"{tt['Reading']:.3f}/{tt['Reasoning']:.3f}/{tt['Decision']:.3f}"
          f"  -> Reasoning>Decision>Reading? {tt['Reasoning'] > tt['Decision'] > tt['Reading']}")
    return np.array([names[l] for l in labels])


subs = complete_subjects()
X, groups = build_matrix(subs)
print(f"\nX={X.shape}  frames={len(groups)}  subjects={len(subs)}\n")

print("=== Q1: STABILITY of current pipeline (sqeuclidean, 10 restarts) across 5 seeds ===")
for seed in range(5):
    report(X, groups, run_once(X, "sqeuclidean", seed), f"[seed {seed}]")

print("\n=== Q2: sqeuclidean vs euclidean (same seed 0) ===")
a = report(X, groups, run_once(X, "sqeuclidean", 0), "[sqeuclidean]")
b = report(X, groups, run_once(X, "euclidean", 0), "[euclidean ]")
print(f"\n   frame-level state agreement = {np.mean(a == b) * 100:.1f}%  "
      f"(how many of the {len(a)} frames get the SAME state under both metrics)")
