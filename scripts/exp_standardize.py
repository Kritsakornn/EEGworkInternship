"""Experiment: does standardizing the 33-dim RSP features before K-medoids give
an SRI that matches the professor's f1/f2/f3 better than the raw-feature pipeline?

Motivated by the thesis (張寶心, sec. discussion) noting "標準化" (standardization)
was applied -- our pipeline feeds raw RSP straight into K-medoids, so high-variance
dimensions (beta) dominate the Euclidean distance. z-scoring each of the 33
dimensions equalises them and changes the partition geometry. This is the one
feature-space transform NOT yet on the ruled-out list (channel-order, band-power,
pooling, epoch-length, filter-order, all-questions, subset-size, clustering-algo,
classifier are all ruled out).

Clusters are FIT on standardized features but NAMED by raw frontal theta
(map_states on raw X), identical to the baseline naming convention, so the only
thing that changes vs baseline is the distance geometry.

Success criterion: mean per-Q acc > 0.581 OR SRI<->f1/f2/f3 max|r| > 0.135.
Otherwise standardization joins the ruled-out list.

Run: python scripts/exp_standardize.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from labels import load_labels
from macrostates import (complete_subjects, frame_complete_subjects,
                         build_matrix, map_states, compute_sri, kmedoids, STATES)
from classify import _per_question

SEED = 0
K = config.N_CLUSTERS
FIT_N = 4000
SIL_N = 5000


def kmedoids_partition(X, rng):
    """Replicate macrostates.cluster_frames partitioning on a given feature
    matrix: subsample, sqeuclidean, N_RESTARTS, keep lowest eq.4 cost, assign all."""
    sample = X[rng.choice(len(X), min(FIT_N, len(X)), replace=False)]
    D = cdist(sample, sample, metric="sqeuclidean")
    best = None
    for _ in range(config.N_RESTARTS):
        medoids = sample[kmedoids(D, K, rng)]
        dist = cdist(X, medoids, metric="sqeuclidean")
        cost = dist.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, dist.argmin(axis=1))
    return best[1]


def max_abs_r(sri, lab):
    """Largest |Pearson r| between any SRI state and any of f1/f2/f3 (same rows)."""
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    rs = [abs(np.corrcoef(m[s], m[f])[0, 1])
          for s in STATES for f in ("f1", "f2", "f3")]
    return max(rs)


def silhouette(X, labels):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), min(SIL_N, len(X)), replace=False)
    return silhouette_score(X[idx], labels[idx], metric="euclidean")


def main():
    lab = load_labels()
    X, groups = build_matrix(frame_complete_subjects(complete_subjects()))
    print(f"feature matrix: {X.shape}  frames={len(groups)}\n")

    Xz = StandardScaler().fit_transform(X)

    # standardized-feature partition; name clusters by RAW frontal theta (baseline convention)
    labels_z = kmedoids_partition(Xz, np.random.default_rng(SEED))
    names_z = map_states(X, labels_z)
    sri_z = compute_sri(labels_z, groups, names_z)

    # baseline for reference (committed K-medoids on raw features)
    sri_km = pd.read_csv(config.TABLE_DIR / "sri.csv")

    print("cluster -> state (standardized):", names_z)
    print(f"\nsilhouette (raw cloud, K-medoids labels reused from std): "
          f"{silhouette(X, labels_z):.3f}")
    print(f"silhouette (standardized cloud):                          "
          f"{silhouette(Xz, labels_z):.3f}")

    res_base = _per_question(sri_km.merge(lab, on=["subject", "q"]), STATES,
                             "BASELINE raw-feature SRI")
    res_z = _per_question(sri_z.merge(lab, on=["subject", "q"]), STATES,
                          "STANDARDIZED-feature SRI")
    _per_question(sri_z.merge(lab, on=["subject", "q"]), ["f1", "f2", "f3"],
                  "Professor f1/f2/f3 (same rows, control)")

    print("\n=== SRI <-> f1/f2/f3 max|Pearson r| ===")
    print(f"  baseline raw SRI:   {max_abs_r(sri_km, lab):.3f}")
    print(f"  standardized SRI:   {max_abs_r(sri_z, lab):.3f}")

    ok = res_z["mean_acc"] > res_base["mean_acc"] or max_abs_r(sri_z, lab) > 0.135
    print(f"\nVERDICT: standardization {'HELPS -- investigate further' if ok else 'does NOT help -> ruled out'}")

    out_dir = config.OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"name": "baseline_raw", **{k: res_base[k] for k in ("mean_acc", "score_r", "score_mae")}},
                  {"name": "standardized", **{k: res_z[k] for k in ("mean_acc", "score_r", "score_mae")}}]
                 ).to_csv(out_dir / "exp_standardize.csv", index=False)
    print(f"saved -> {out_dir / 'exp_standardize.csv'}")


if __name__ == "__main__":
    main()
