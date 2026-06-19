"""Experiment B: does a different Layer-1 clustering method give an SRI that
predicts answer-correctness better than K-medoids?

K-medoids is the bottleneck: our SRI predicts correctness at ~chance while the
professor's f1/f2/f3 reach 0.79. This swaps ONLY the frame-clustering step and
re-runs the identical downstream (frontal-theta state naming -> SRI -> the
paper's nine per-question logistic models). The 33-dim RSP feature matrix is
built once and shared across methods, so the only thing that changes is how the
frames are partitioned into three macrostates.

Methods (all K=3, mapped to Reading/Reasoning/Decision by frontal theta):
  - kmedoids   : current pipeline baseline (loaded from sri.csv)
  - kmeans     : centroid, squared-euclidean
  - gmm_hard   : Gaussian mixture, argmax assignment
  - gmm_soft   : Gaussian mixture, posterior-weighted SRI (soft proportions)
  - agglomerative : Ward linkage on a 4000-frame subsample, full frames assigned
                    to the nearest sub-cluster centroid (out-of-sample extension)

Writes outputs/experiments/exp_clustering_compare.csv.
Run: python scripts/exp_clustering_compare.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from labels import load_labels
from macrostates import (complete_subjects, frame_complete_subjects,
                         build_matrix, map_states, compute_sri, STATES)
from classify import _per_question

SEED = 0
K = config.N_CLUSTERS
SUBSAMPLE = 4000  # same fit size as the K-medoids pipeline


def soft_sri(proba, groups, names):
    """Posterior-weighted SRI: mean cluster-membership probability per (subj, q),
    relabelled to states. Rows still sum to ~1 because posteriors do."""
    cols = [names[i] for i in range(proba.shape[1])]
    df = pd.DataFrame(proba, columns=cols)
    df["subject"] = [g[0] for g in groups]
    df["q"] = [g[1] for g in groups]
    return df.groupby(["subject", "q"], as_index=False)[STATES].mean()


def evaluate(sri, lab, name):
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"
    res = _per_question(m, STATES, name)
    return {"method": name, "mean_acc": res["mean_acc"],
            "score_r": res["score_r"], "score_mae": res["score_mae"]}


def main():
    lab = load_labels()
    X, groups = build_matrix(frame_complete_subjects(complete_subjects()))
    print(f"feature matrix: {X.shape}  frames={len(groups)}\n")
    rng = np.random.default_rng(SEED)
    rows = []

    # baseline: the committed K-medoids partition (sri.csv is its SRI on this X)
    sri_km = pd.read_csv(config.TABLE_DIR / "sri.csv")
    rows.append(evaluate(sri_km, lab, "kmedoids (baseline)"))

    # kmeans
    km = KMeans(K, n_init=10, random_state=SEED).fit(X)
    rows.append(evaluate(compute_sri(km.labels_, groups, map_states(X, km.labels_)),
                         lab, "kmeans"))

    # gaussian mixture: hard (argmax) and soft (posterior-weighted)
    gm = GaussianMixture(K, covariance_type="full", n_init=5,
                         random_state=SEED).fit(X)
    hard = gm.predict(X)
    names = map_states(X, hard)
    rows.append(evaluate(compute_sri(hard, groups, names), lab, "gmm_hard"))
    rows.append(evaluate(soft_sri(gm.predict_proba(X), groups, names), lab, "gmm_soft"))

    # agglomerative (Ward) on a subsample, full frames -> nearest sub-centroid
    idx = rng.choice(len(X), min(SUBSAMPLE, len(X)), replace=False)
    sub_lab = AgglomerativeClustering(K, linkage="ward").fit_predict(X[idx])
    centroids = np.stack([X[idx][sub_lab == j].mean(axis=0) for j in range(K)])
    agg = cdist(X, centroids, metric="sqeuclidean").argmin(axis=1)
    rows.append(evaluate(compute_sri(agg, groups, map_states(X, agg)),
                         lab, "agglomerative"))

    out_dir = config.OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = out_dir / "exp_clustering_compare.csv"
    df.to_csv(out, index=False)

    print("\n=== summary (our SRI, by clustering method; baseline acc 0.581) ===")
    print(df.sort_values("mean_acc", ascending=False).to_string(index=False))
    print("\n(reference: professor f1/f2/f3 reach mean_acc 0.788)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
