"""Check whether the 'UNIQUE' cuteeg files are actually re-cuts of the same data
that lives in EEG_All_Questions / EEG_All_Fixations.

A file can be 'UNIQUE' (not byte-identical to any single other file) yet still be
built entirely from samples of the all-questions / all-fixations recording — e.g.
a shorter slice, or several fixation windows concatenated. We measure that with:

  row-overlap%   : fraction of the cuteeg file's time-samples (rows) that appear
                   exactly in the candidate file  (robust to concatenation)
  contig offset  : if overlap ~100%, is it ONE continuous slice B[t:t+n] == A ?

Candidates compared per file = the SAME question number Q{n} in:
    EEG_All_Questions/iStarCutEEGdata_Q{n}.mat
    EEG_All_Fixations/CutEEGdata_Q{n}_AllFixations.mat   (new, Mar 2023)
    EEG_All_Fixations/CutEEGdata_Q{n}_AllFixaitons.mat   (old typo, Jan 2023)

Reads which files are UNIQUE from outputs/reports/cuteeg_provenance_table.txt.

Usage:
    python scripts/check_unique_subsegment.py S13     # one subject (verbose)
    python scripts/check_unique_subsegment.py         # all subjects -> summary + report file
"""
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "iStar_data"
PROV = ROOT / "outputs" / "reports" / "cuteeg_provenance_table.txt"


def arr(p):
    m = sio.loadmat(p)
    ks = [k for k in m if not k.startswith("__")]
    a = np.asarray(max((np.asarray(m[k]) for k in ks), key=lambda x: x.size), dtype=float)
    if a.ndim == 2 and a.shape[0] < a.shape[1]:
        a = a.T  # orient to (samples, channels)
    return a


def row_overlap(A, B):
    """Fraction of A's rows that appear exactly as a row of B."""
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        return 0.0
    bset = {r.tobytes() for r in np.ascontiguousarray(B)}
    hit = sum(r.tobytes() in bset for r in np.ascontiguousarray(A))
    return hit / len(A)


def contig_offset(A, B):
    """Return t if A == B[t:t+len(A)] exactly, else None."""
    nA, nB = len(A), len(B)
    if A.shape[1] != B.shape[1] or nA > nB:
        return None
    cand = np.where((B[: nB - nA + 1] == A[0]).all(axis=1))[0]
    for t in cand:
        if np.array_equal(B[t : t + nA], A):
            return int(t)
    return None


def candidates(subj, qn):
    base = DATA / subj / "CutEEG_Cognitive"
    return {
        "All_Questions": base / "EEG_All_Questions" / f"iStarCutEEGdata_Q{qn}.mat",
        "AllFix_new": base / "EEG_All_Fixations" / f"CutEEGdata_Q{qn}_AllFixations.mat",
        "AllFix_old": base / "EEG_All_Fixations" / f"CutEEGdata_Q{qn}_AllFixaitons.mat",
    }


def check_file(subj, fname):
    qn = re.search(r"Q(\d+)", fname).group(1)
    A = arr(DATA / subj / "CutEEG_Cognitive" / fname)
    best = {"label": "(no candidate)", "overlap": 0.0, "bshape": None, "offset": None}
    for label, p in candidates(subj, qn).items():
        if not p.exists():
            continue
        f = row_overlap(A, arr(p))
        if f > best["overlap"]:
            B = arr(p)
            best = {
                "label": label,
                "overlap": f,
                "bshape": B.shape,
                "offset": contig_offset(A, B) if f > 0.999 else None,
            }
    return qn, A.shape, best


def unique_files():
    rows = []
    for line in PROV.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 4 and parts[3].startswith("UNIQUE"):
            rows.append((parts[0], parts[1]))
    return rows


def classify(overlap):
    if overlap > 0.999:
        return "FULL (100% — same data, re-cut)"
    if overlap >= 0.05:
        return "PARTIAL"
    return "INDEPENDENT (~0%)"


def main():
    targets = unique_files()
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
        targets = [(s, f) for s, f in targets if s in only]

    lines = ["subject\tfile\tshape\tbest_match\toverlap%\tcontiguous\tclass"]
    counts = {"FULL (100% — same data, re-cut)": 0, "PARTIAL": 0, "INDEPENDENT (~0%)": 0}
    verbose = len(sys.argv) > 1

    cur = None
    for subj, fname in targets:
        qn, ashape, b = check_file(subj, fname)
        cls = classify(b["overlap"])
        counts[cls] += 1
        contig = "yes@%d" % b["offset"] if b["offset"] is not None else "no/concat"
        lines.append(
            f"{subj}\t{fname}\t{ashape}\t{b['label']}\t{b['overlap']*100:.1f}\t{contig}\t{cls}"
        )
        if verbose:
            if subj != cur:
                print(f"\n=== {subj} ===")
                cur = subj
            print(
                f"  {fname:22s} {str(ashape):14s} -> {b['label']:13s} "
                f"vs {str(b['bshape']):14s} overlap={b['overlap']*100:5.1f}% "
                f"contig={contig:10s} {cls}"
            )

    out = ROOT / "outputs" / "reports" / "unique_subsegment_report.txt"
    out.write_text("\n".join(lines) + "\n")

    total = sum(counts.values())
    print(f"\n===== UNIQUE files analysed: {total} =====")
    for k, v in counts.items():
        pct = (v / total * 100) if total else 0
        print(f"  {k:34s}: {v:4d}  ({pct:4.1f}%)")
    print(f"\nPer-file report written to: {out}")


if __name__ == "__main__":
    main()
