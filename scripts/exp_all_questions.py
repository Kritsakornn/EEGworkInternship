"""Experiment: would using EEG_All_Questions (the raw, full-question recording)
instead of CutEEG_Cognitive give an SRI that predicts answer-correctness better?

CutEEG_Cognitive/EEG_All_Questions/iStarCutEEGdata_Q{q}.mat holds ONE file per
question covering the WHOLE question period (~100 s, raw: ~83% power >30 Hz,
amplitude up to +/-130000 uV), versus CutEEGdata_Q{q}_{seg}.mat which is the
filtered, cognitive-only cut (~45 s, ~2% power >30 Hz). Our loader re-applies
the paper's 1-30 Hz band-pass + 60 Hz notch either way, so the real variable
tested here is "full question period" vs "cognitive segment only".

Mirrors the committed Layer-1 (same _bandpass_notch -> rsp_features -> K-medoids
seed=0 / 10 restarts -> frontal-theta naming -> SRI) and the committed Layer-2
(classify._per_question). Compares OUR All-Questions SRI and the professor's
f1/f2/f3 on the same rows. Reads from the dataset only; writes nothing to
outputs/tables. Baseline to beat: cognitive-cut SRI mean per-Q acc 0.581.

Run: python scripts/exp_all_questions.py
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from features import _bandpass_notch, rsp_features
from macrostates import map_states, compute_sri, kmedoids, STATES
from labels import load_labels
from classify import _per_question

ALLQ = "EEG_All_Questions"


def allq_file(subject, q):
    f = config.DATA_DIR / subject / "CutEEG_Cognitive" / ALLQ / f"iStarCutEEGdata_Q{q}.mat"
    return f if f.exists() else None


def load_allq(subject, q):
    m = sio.loadmat(allq_file(subject, q))
    key = [k for k in m if not k.startswith("__")][0]
    return _bandpass_notch(np.asarray(m[key], dtype=float))  # same filter as pipeline


def complete_allq_subjects():
    """Subjects whose Q1-Q9 All-Questions files all exist AND keep >=1 frame
    after the same +/-150 uV artifact rejection used by the main pipeline."""
    keep = []
    for d in sorted(config.DATA_DIR.glob("S*"), key=lambda p: int(p.name[1:])):
        s = d.name
        if not all(allq_file(s, q) for q in config.VISUAL_QUESTIONS):
            continue
        if all(len(rsp_features(load_allq(s, q))) for q in config.VISUAL_QUESTIONS):
            keep.append(s)
    return keep


def main():
    subjects = complete_allq_subjects()
    print(f"All-Questions subjects complete + surviving rejection: {len(subjects)}")
    print(" ", subjects)

    feats, groups = [], []
    for s in subjects:
        for q in config.VISUAL_QUESTIONS:
            x = rsp_features(load_allq(s, q))
            feats.append(x)
            groups += [(s, q)] * len(x)
    X = np.concatenate(feats)
    print(f"feature matrix: {X.shape}  frames={len(groups)}")

    # --- clustering core, mirrors macrostates.cluster_frames (seed 0, 10 restarts) ---
    rng = np.random.default_rng(0)
    sample = X[rng.choice(len(X), min(4000, len(X)), replace=False)]
    D = cdist(sample, sample, metric="sqeuclidean")
    best = None
    for _ in range(config.N_RESTARTS):
        medoids = sample[kmedoids(D, config.N_CLUSTERS, rng)]
        dist = cdist(X, medoids, metric="sqeuclidean")
        cost = dist.min(axis=1).sum()
        if best is None or cost < best[0]:
            best = (cost, dist.argmin(axis=1))
    labels = best[1]
    sri = compute_sri(labels, groups, map_states(X, labels))

    lab = load_labels()
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"
    print(f"\nmerged: {len(m)} rows, {m.subject.nunique()} subjects\n")

    _per_question(m, STATES, "All-Questions SRI")
    _per_question(m, ["f1", "f2", "f3"], "Professor f1/f2/f3 (same rows)")
    print("\n(cognitive-cut baseline: OUR SRI 0.581 / professor 0.788)")


if __name__ == "__main__":
    main()
