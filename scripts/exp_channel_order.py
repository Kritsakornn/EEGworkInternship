"""Diagnostic: does a different 19-channel column order make OUR SRI correlate
with the professor's f1/f2/f3 (and predict correctness)? If yes, the assumed
standard-10-20 order is the bug; if no, the gap is K-medoids instability or a
different method on the professor's side.

For each candidate column permutation we re-featurise the 22 complete subjects,
re-cluster (fast settings: 2000-frame fit, 3 restarts), build SRI, then report:
  - best-matching mean |Pearson r| between our 3 state ratios and {f1,f2,f3}
    (tries all 6 state<->f assignments, since the professor's order is unknown)
  - Layer-2 AUC (5-fold GroupKFold logistic regression) on that SRI

Note: our 11 task channels are symmetric L/R pairs + midline, so a pure
left-right swap is a no-op for K-medoids (permutation-invariant) -- lr_swap is
included only to confirm this empirically.

Run: python scripts/exp_channel_order.py
"""
import sys
from pathlib import Path
from itertools import permutations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import macrostates as M
from features import load_question_eeg, rsp_features
from labels import load_labels

STATES = ["Reading", "Reasoning", "Decision"]

# candidate column orders (index into the 19 physical columns)
IDENT = list(range(19))
# left-right pairs in CHANNELS_19 order: Fp1/Fp2 F7/F8 F3/F4 T3/T4 C3/C4 T5/T6 P3/P4 O1/O2
LR_SWAP = [1, 0, 6, 5, 4, 3, 2, 11, 10, 9, 8, 7, 16, 15, 14, 13, 12, 18, 17]
REVERSE = list(range(18, -1, -1))


def featurize(perm, subjects):
    """Build (X, groups) with the 19 columns permuted before channel selection."""
    feats, groups = [], []
    for s in subjects:
        for q in config.VISUAL_QUESTIONS:
            eeg = load_question_eeg(s, q)[:, perm]
            x = rsp_features(eeg)
            feats.append(x)
            groups += [(s, q)] * len(x)
    return np.concatenate(feats), groups


def recluster(X, seed, fit_n=2000, restarts=3):
    """Fast K-medoids (exploration settings) -> per-frame labels."""
    rng = np.random.default_rng(seed)
    sample = X[rng.choice(len(X), min(fit_n, len(X)), replace=False)]
    D = cdist(sample, sample, "sqeuclidean")
    best = None
    for _ in range(restarts):
        med = sample[M.kmedoids(D, config.N_CLUSTERS, rng)]
        dist = cdist(X, med, "sqeuclidean")
        cost = dist.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, dist.argmin(axis=1))
    return best[1]


def sri_df(X, groups, seed):
    labels = recluster(X, seed)
    names = M.map_states(X, labels)
    return M.compute_sri(labels, groups, names)


def best_match_corr(ours, prof):
    """Max over the 6 state<->f assignments of the mean |Pearson r|."""
    best = -1.0
    for p in permutations(range(3)):
        rs = [abs(np.corrcoef(ours[:, i], prof[:, p[i]])[0, 1]) for i in range(3)]
        best = max(best, float(np.mean(rs)))
    return best


def layer2_auc(merged):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    proba = cross_val_predict(clf, merged[STATES].values, merged.label.values,
                              cv=GroupKFold(5), groups=merged.subject.values,
                              method="predict_proba")[:, 1]
    return roc_auc_score(merged.label.values, proba)


def score(name, X, groups, lab, seed):
    sri = sri_df(X, groups, seed)
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    corr = best_match_corr(m[STATES].values, m[["f1", "f2", "f3"]].values)
    auc = layer2_auc(m)
    print(f"  {name:16s} seed={seed}  best|r|={corr:.3f}  AUC={auc:.3f}")
    return corr, auc


def main():
    lab = load_labels()
    subjects = M.frame_complete_subjects(M.complete_subjects())
    print(f"subjects={len(subjects)}  (professor AUC reference = 0.850)\n")

    print("IDENTITY across seeds (K-medoids instability check):")
    Xi, gi = featurize(IDENT, subjects)
    for seed in range(5):
        score("identity", Xi, gi, lab, seed)

    print("\nALTERNATIVE channel orders (seed 0):")
    for name, perm in [("lr_swap", LR_SWAP), ("reverse", REVERSE)]:
        X, g = featurize(perm, subjects)
        score(name, X, g, lab, 0)
    rng = np.random.default_rng(7)
    for k in range(2):
        perm = list(rng.permutation(19))
        X, g = featurize(perm, subjects)
        score(f"random{k+1}", X, g, lab, 0)


if __name__ == "__main__":
    main()
