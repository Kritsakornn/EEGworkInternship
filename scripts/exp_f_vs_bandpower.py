"""Diagnostic: are the professor's f1/f2/f3 the mean theta/alpha/beta band
ratios (RSP), rather than K-medoids macrostate ratios?

OUR macrostate SRI is uncorrelated with f1/f2/f3 (max|r|~0.08). The open
question (see docs/memory) is what f1/f2/f3 actually are. f1+f2+f3 sum to ~100,
so they are 3-part proportions -- consistent with EITHER 3 macrostate ratios OR
the 3 band ratios theta/alpha/beta. This script settles it: for every complete
(subject, question) it averages OUR RSP over all surviving frames and the 11
task channels -> 3 numbers (mean theta, alpha, beta RSP), then correlates those
against f1/f2/f3 and compares value-for-value under the best assignment.

  - high |r| (e.g. >0.6) and small value gap  => f1/f2/f3 ARE band ratios
    (our whole Layer-1 K-medoids step solves the wrong target)
  - all |r| ~0 (like the macrostate SRI)       => f1/f2/f3 are something else

The professor's f-order is unknown, so all 6 band<->f assignments are tried and
the best (max mean |r|) is reported, exactly as scripts/exp_channel_order.py
does for the SRI<->f comparison.

Run: python scripts/exp_f_vs_bandpower.py
"""
import sys
from pathlib import Path
from itertools import permutations

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
import macrostates as M
from labels import load_labels

STATES = ["Reading", "Reasoning", "Decision"]
BANDS = ["theta", "alpha", "beta"]


def band_means(subjects):
    """Mean RSP theta/alpha/beta per (subject, q), over all frames and 11 channels.

    Reuses macrostates.build_matrix so the frames are EXACTLY those that feed the
    SRI clustering (same artifact-drop), keeping this comparison apples-to-apples.
    """
    X, groups = M.build_matrix(subjects)            # X (N, 33), groups [(s, q)]*N
    df = pd.DataFrame(X)
    df["subject"] = [s for s, _ in groups]
    df["q"] = [q for _, q in groups]
    g = df.groupby(["subject", "q"], sort=False).mean()        # (n_sq, 33)
    # 33 dims = [ch0 theta, ch0 alpha, ch0 beta, ch1 theta, ...] -> collapse 11 ch
    arr = g.values.reshape(len(g), 11, 3).mean(axis=1)         # (n_sq, 3)
    return pd.DataFrame(arr, columns=BANDS, index=g.index).reset_index()


def best_assignment(ours, prof):
    """Over the 6 col<->f assignments, return (best mean|r|, perm, per-pair |r|)."""
    best = (-1.0, None, None)
    for p in permutations(range(3)):
        rs = [abs(np.corrcoef(ours[:, i], prof[:, p[i]])[0, 1]) for i in range(3)]
        if np.mean(rs) > best[0]:
            best = (float(np.mean(rs)), p, rs)
    return best


def main():
    lab = load_labels()
    subjects = M.frame_complete_subjects(M.complete_subjects())
    print(f"subjects={len(subjects)}  (band-power vs professor f1/f2/f3)\n")

    bm = band_means(subjects)
    m = bm.merge(lab, on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"
    print(f"merged rows: {len(m)}  subjects: {m.subject.nunique()}\n")

    ours, prof = m[BANDS].values, m[["f1", "f2", "f3"]].values

    C = np.array([[np.corrcoef(ours[:, i], prof[:, j])[0, 1] for j in range(3)]
                  for i in range(3)])
    print("Pearson corr (rows = our mean RSP theta/alpha/beta, cols = f1/f2/f3):")
    print(pd.DataFrame(np.round(C, 3), index=BANDS,
                       columns=["f1", "f2", "f3"]).to_string())

    score, perm, rs = best_assignment(ours, prof)
    pairing = {BANDS[i]: f"f{perm[i] + 1}" for i in range(3)}
    print(f"\nbest band<->f assignment: {pairing}")
    print(f"  per-pair |r| = {[round(x, 3) for x in rs]}   mean|r| = {score:.3f}")
    print(f"  max single |r| = {np.abs(C).max():.3f}")

    # value comparison under the best assignment: if f1/f2/f3 ARE the band ratios
    # (as percentages), then f/100 should approximately EQUAL our band RSP.
    f_frac = prof[:, list(perm)] / 100.0
    mae = np.abs(f_frac - ours).mean(axis=0)
    print("\nvalue check under best assignment (f/100 vs our band RSP):")
    for i, b in enumerate(BANDS):
        print(f"  {b:5s}: our mean={ours[:, i].mean():.3f}  "
              f"prof {pairing[b]}/100 mean={f_frac[:, i].mean():.3f}  "
              f"MAE={mae[i]:.3f}")

    # benchmark: the macrostate SRI from the committed pipeline, same metric
    sri = pd.read_csv(config.TABLE_DIR / "sri.csv")
    ms = sri.merge(lab, on=["subject", "q"], how="inner")
    sri_score = best_assignment(ms[STATES].values,
                                ms[["f1", "f2", "f3"]].values)[0]
    print(f"\nbenchmark macrostate SRI vs f1/f2/f3: best mean|r| = {sri_score:.3f}")
    print("\nINTERPRETATION: band-power explains f1/f2/f3 well only if its mean|r| "
          "is high (>~0.6) AND clearly beats the macrostate SRI benchmark above.")


if __name__ == "__main__":
    main()
