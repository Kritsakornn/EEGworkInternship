"""Experiment: does the LITERAL reading of the paper's "sliding overlap of 0.1 s"
make OUR Layer-1 SRI match the professor's f1/f2/f3?

Our committed pipeline reads "overlap 0.1 s" as STEP = 0.1 s (95% overlap, ~50k
highly-autocorrelated frames). The literal reading is OVERLAP = 0.1 s on a 2 s
window -> STEP = 1.9 s, i.e. near-independent ~2.6k frames. As a side effect the
artifact drop nearly vanishes (frames barely overlap, so one spike no longer
kills 20 frames), so more subjects survive complete-9 -- the control f1/f2/f3 is
therefore evaluated on the SAME rows for a fair comparison.

Only config.STEP_SEC changes (WIN_SEC stays 2.0); everything downstream
(K-medoids seed=0/10 restarts -> frontal-theta naming -> SRI -> per-question
Layer-2) is identical. Writes a separate sri_overlap_literal.csv; does NOT touch
the committed 2 s/0.1 s sri.csv.

Baseline to beat (2 s/0.1 s, 22 subj): OUR SRI mean per-Q acc 0.581 / score r
0.135 ; professor f1/f2/f3 0.788 / 0.741 ; SRI<->f best|r| ~0.06.
Success: OUR mean per-Q acc rises toward ~0.79 AND best|r| clearly above ~0.06.

Run: python scripts/exp_overlap_literal.py
"""
import sys
from pathlib import Path
from itertools import permutations

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
config.STEP_SEC = 1.9  # literal "overlap 0.1 s" on a 2 s window -> step 1.9 s

import macrostates as M
from labels import load_labels
from classify import _per_question, STATES


def best_match_corr(ours, prof):
    """Max over the 6 state<->f assignments of the mean |Pearson r|."""
    return max(float(np.mean([abs(np.corrcoef(ours[:, i], prof[:, p[i]])[0, 1])
                              for i in range(3)]))
               for p in permutations(range(3)))


def main():
    print(f"=== literal overlap 0.1 s "
          f"(WIN_SEC={config.WIN_SEC} STEP_SEC={config.STEP_SEC} -> "
          f"overlap {config.WIN_SEC - config.STEP_SEC:.1f} s) ===")
    X, groups, labels, names = M.cluster_frames()
    sri = M.compute_sri(labels, groups, names)
    out_dir = config.OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sri_overlap_literal.csv"
    sri.to_csv(out, index=False)
    print(f"saved -> {out}")

    lab = load_labels()
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"
    print(f"\nmerged: {len(m)} rows, {m.subject.nunique()} subjects, "
          f"{m.q.nunique()} questions")

    _per_question(m, STATES, "OUR SRI (overlap 0.1 s / step 1.9 s)")
    _per_question(m, ["f1", "f2", "f3"], "Professor f1/f2/f3 (same rows)")

    corr = best_match_corr(m[STATES].values, m[["f1", "f2", "f3"]].values)
    print(f"\nSRI(step 1.9 s) <-> f1/f2/f3 best mean|r| = {corr:.3f}   "
          f"(2 s/0.1 s baseline ~0.06; chance ~0)")

    print("\nVERDICT: helps only if OUR mean per-Q acc rises toward ~0.79 AND "
          "best|r| rises clearly above ~0.06.")


if __name__ == "__main__":
    main()
