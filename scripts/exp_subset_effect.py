"""THROWAWAY experiment (safe to delete): is the unclear clustering caused by the
professor's complete-9 SUBSET restriction, or is it inherent to our data?

Compares K=3 silhouette on:
  (A) complete-9 subjects only (27)            <- what the main pipeline uses
  (B) ALL available subjects/questions we have <- every cognitive .mat in our copy

CAVEAT: even (B) is NOT the paper's full dataset. The paper had all 40 subjects with
complete data, a 3-min resting baseline, and artifact-FREE segments; our .mat copy is
incomplete, baseline-less, and amplitude-clipped. So this bounds the SUBSET-SIZE effect
only -- it cannot rule out that the paper's fuller/cleaner data clusters better.

Run:  PYTHONPATH=src python3 scripts/exp_subset_effect.py
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

import config
from features import find_cognitive_files, load_question_eeg, rsp_features
from macrostates import complete_subjects, build_matrix, kmedoids

SUB_N = 5000


def k3_silhouette(X):
    """K=3 K-medoids (lowest of N_RESTARTS) on a subsample -> silhouette (euclidean)."""
    rng = np.random.default_rng(0)
    S = X[rng.choice(len(X), min(SUB_N, len(X)), replace=False)]
    D = cdist(S, S, metric="sqeuclidean")
    best = None
    for _ in range(config.N_RESTARTS):
        dist = D[:, kmedoids(D, 3, rng)]
        cost = dist.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, dist.argmin(axis=1))
    return silhouette_score(S, best[1], metric="euclidean")


def build_all_available():
    """Pool RSP frames from EVERY subject/question that has a cognitive .mat."""
    feats, subs = [], []
    for d in sorted(config.DATA_DIR.glob("S*"), key=lambda p: int(p.name[1:])):
        got = False
        for q in config.VISUAL_QUESTIONS:
            if find_cognitive_files(d.name, q):
                feats.append(rsp_features(load_question_eeg(d.name, q)))
                got = True
        if got:
            subs.append(d.name)
    return np.concatenate(feats), subs


cs = complete_subjects()
Xc, _ = build_matrix(cs)
print(f"(A) complete-9   : {len(cs)} subjects, {len(Xc)} frames")
silA = k3_silhouette(Xc)
print(f"    K=3 silhouette = {silA:.3f}")

Xa, asubs = build_all_available()
print(f"(B) all-available: {len(asubs)} subjects, {len(Xa)} frames")
silB = k3_silhouette(Xa)
print(f"    K=3 silhouette = {silB:.3f}")

print(f"\nΔ(all - complete9) = {silB - silA:+.3f}")
print("-> Both still <0.25: the SUBSET is not the cause; clustering is unclear in our"
      " data either way (but our data != paper's full/clean data)."
      if max(silA, silB) < 0.25 else
      "-> all-available is clearly higher: the complete-9 subset DID blur the clusters.")
