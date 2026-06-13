"""Steps 4-5: discover 3 cognitive macrostates with K-medoids (pooled across
all subjects/questions) and compute the State Ratio Index (SRI) per question.

K-medoids is fit on a random subsample of frames for tractability, then every
frame is assigned to its nearest medoid (paper eq. 4). Clusters are mapped to
Reading/Reasoning/Decision by frontal theta power (Reasoning > Decision >
Reading), a heuristic to validate later against topographic maps.

Run: python src/macrostates.py
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import config
from features import find_cognitive_files, load_question_eeg, rsp_features

FRONTAL = ["Fp1", "Fp2", "F3", "Fz", "F4"]
STATES = ["Reading", "Reasoning", "Decision"]


def kmedoids(D, k, rng, max_iter=100):
    """Alternating K-medoids on a square distance matrix -> medoid indices."""
    medoids = rng.choice(len(D), k, replace=False)
    for _ in range(max_iter):
        labels = D[:, medoids].argmin(axis=1)
        new = medoids.copy()
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members):
                new[j] = members[D[np.ix_(members, members)].sum(axis=1).argmin()]
        if set(new) == set(medoids):
            break
        medoids = new
    return medoids


def complete_subjects():
    """Subjects that have cognitive files for every visual question Q1-Q9."""
    subs = []
    for d in sorted(config.DATA_DIR.glob("S*"), key=lambda p: int(p.name[1:])):
        if all(find_cognitive_files(d.name, q) for q in config.VISUAL_QUESTIONS):
            subs.append(d.name)
    return subs


def build_matrix(subjects):
    """Stack RSP frames from all (subject, question) -> X, and a group label list."""
    feats, groups = [], []
    for s in subjects:
        for q in config.VISUAL_QUESTIONS:
            x = rsp_features(load_question_eeg(s, q))
            feats.append(x)
            groups += [(s, q)] * len(x)
    return np.concatenate(feats), groups


def _frontal_theta_idx():
    """Indices in the 33-dim vector for frontal-channel theta power."""
    return [config.CHANNELS_11.index(c) * 3 + 0 for c in FRONTAL]  # theta offset = 0


def map_states(X, labels):
    """Order clusters by mean frontal theta: lowest=Reading, highest=Reasoning."""
    idx = _frontal_theta_idx()
    order = sorted(range(config.N_CLUSTERS),
                   key=lambda c: X[labels == c][:, idx].mean())
    return {order[0]: "Reading", order[1]: "Decision", order[2]: "Reasoning"}


def compute_sri(labels, groups, names):
    df = pd.DataFrame(groups, columns=["subject", "q"])
    df["state"] = [names[l] for l in labels]
    counts = df.groupby(["subject", "q", "state"]).size().unstack(fill_value=0)
    for s in STATES:
        if s not in counts:
            counts[s] = 0
    sri = counts[STATES].div(counts[STATES].sum(axis=1), axis=0)
    return sri.reset_index()


def cluster_frames():
    """Run the full Layer-1 pipeline. Returns (X, groups, labels, names)."""
    subjects = complete_subjects()
    print(f"complete subjects (Q1-Q9): {len(subjects)}")
    print(" ", subjects)

    X, groups = build_matrix(subjects)
    print(f"feature matrix: {X.shape}  frames={len(groups)}")

    rng = np.random.default_rng(0)
    fit_n = min(4000, len(X))
    sample = X[rng.choice(len(X), fit_n, replace=False)]
    D = cdist(sample, sample, metric="sqeuclidean")          # 4000x4000
    # K-medoids has many local optima on this data, so restart N times and keep
    # the assignment with the lowest total cost (paper eq.4, over all frames).
    best = None  # (cost, labels) of the lowest-cost restart
    for _ in range(config.N_RESTARTS):
        medoids = sample[kmedoids(D, config.N_CLUSTERS, rng)]
        dist = cdist(X, medoids, metric="sqeuclidean")       # eq.(4): assign all
        cost = dist.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, dist.argmin(axis=1))
    labels = best[1]

    names = map_states(X, labels)
    print("cluster -> state:", names)
    return X, groups, labels, names


def main():
    _, groups, labels, names = cluster_frames()
    sri = compute_sri(labels, groups, names)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "sri.csv"
    sri.to_csv(out, index=False)

    print(sri.head(12).to_string(index=False))
    chk = sri[STATES].sum(axis=1)
    print(f"SRI row sums: min={chk.min():.4f} max={chk.max():.4f} (expect 1.0)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
