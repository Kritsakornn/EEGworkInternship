"""Experiment: does 1-second NON-overlapping epoching (thesis: 每秒一段) make
OUR Layer-1 SRI predict answer-correctness, closing the gap to the professor's
f1/f2/f3?

The committed pipeline uses a 2 s STFT window / 0.1 s step (95% overlap -> highly
autocorrelated frames); the thesis (張寶心 ...docx) says epochs were cut into
1-second segments. This is the top untested lever for the Layer-1 partition
mismatch (see docs/MEMORY.md "Next session 2026-06-20").

It overrides config.WIN_SEC/STEP_SEC to 1.0/1.0 and re-runs the FULL Layer-1
pipeline (same K-medoids seed=0 / 10 restarts), then the per-question Layer-2
models (reusing classify._per_question) on OUR 1 s-SRI and, on the SAME rows,
the professor's f1/f2/f3 as reference. Writes a separate sri_1s.csv -- it does
NOT touch the committed 2 s sri.csv.

Baseline to beat (2 s pipeline, 22 subj): OUR SRI mean per-Q acc 0.581 /
score r 0.135 ; professor f1/f2/f3 0.788 / 0.741 ; SRI<->f max|r| ~0.06.

Run: python scripts/exp_epoch_1s.py
"""
import sys
from pathlib import Path
from itertools import permutations

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
config.WIN_SEC = 1.0   # thesis: 1-second epochs (每秒一段)
config.STEP_SEC = 1.0  # non-overlapping

import macrostates as M
from labels import load_labels
from classify import _per_question, STATES


def best_match_corr(ours, prof):
    """Max over the 6 state<->f assignments of the mean |Pearson r|."""
    return max(float(np.mean([abs(np.corrcoef(ours[:, i], prof[:, p[i]])[0, 1])
                              for i in range(3)]))
               for p in permutations(range(3)))


def main():
    print(f"=== 1-second non-overlapping epochs "
          f"(WIN_SEC={config.WIN_SEC} STEP_SEC={config.STEP_SEC}) ===")
    X, groups, labels, names = M.cluster_frames()
    sri = M.compute_sri(labels, groups, names)
    out_dir = config.OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sri_1s.csv"
    sri.to_csv(out, index=False)
    print(f"saved -> {out}")

    lab = load_labels()
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"
    print(f"\nmerged: {len(m)} rows, {m.subject.nunique()} subjects, "
          f"{m.q.nunique()} questions")

    _per_question(m, STATES, "OUR SRI (1 s epochs)")
    _per_question(m, ["f1", "f2", "f3"], "Professor f1/f2/f3 (same rows)")

    corr = best_match_corr(m[STATES].values, m[["f1", "f2", "f3"]].values)
    print(f"\nSRI(1 s) <-> f1/f2/f3 best mean|r| = {corr:.3f}   "
          f"(2 s baseline ~0.06; chance ~0)")

    print("\nVERDICT: 1 s epochs help only if OUR mean per-Q acc rises toward "
          "~0.79 AND best|r| rises clearly above ~0.06.")


if __name__ == "__main__":
    main()
