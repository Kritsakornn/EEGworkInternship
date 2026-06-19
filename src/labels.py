"""Load the professor's answer-correctness labels (the Layer-2 target).

label_answer_data/0{q}total.xlsx holds, per question q, one row per subject:
    subject, question, f1, f2, f3, label
f1/f2/f3 are the professor's own macrostate ratios (%), label is 1=correct /
0=incorrect. The sheets carry ~1e6 trailing blank Excel rows and one stray
non-binary label (0.475), so we keep only rows with a valid subject id (s\\d+)
and a 0/1 label.

Run: python src/labels.py
"""
import pandas as pd

import config

LABEL_DIR = config.DATA_DIR / "label_answer_data"


def load_labels():
    """-> cleaned DataFrame [subject (S1..S40), q, f1, f2, f3, label]."""
    frames = []
    for f in sorted(LABEL_DIR.glob("[0-9][0-9]total.xlsx")):
        q = int(f.name[:2])
        d = pd.read_excel(f)[["subject", "f1", "f2", "f3", "label"]]
        d = d[d.subject.astype(str).str.match(r"^s\d+$", case=False, na=False)]  # real rows (any case)
        d = d[d.label.isin([0, 1])]                                  # drop NaN + 0.475
        d["q"] = q
        d["subject"] = d.subject.str.upper()                         # s1 -> S1
        frames.append(d)
    lab = pd.concat(frames, ignore_index=True)
    lab["label"] = lab.label.astype(int)
    # a (subject, q) must be unique; a duplicate means a question file was read
    # twice or lists a subject twice -> would fan out the Layer-2 merge silently
    assert not lab.duplicated(["subject", "q"]).any(), \
        "duplicate (subject, q) in label files"
    return lab[["subject", "q", "f1", "f2", "f3", "label"]]


def main():
    lab = load_labels()
    print(f"clean labels: {len(lab)} rows, {lab.subject.nunique()} subjects, "
          f"Q{sorted(lab.q.unique())}")
    print(f"balance: correct={int((lab.label == 1).sum())} "
          f"wrong={int((lab.label == 0).sum())}")
    print(lab.head().to_string(index=False))


if __name__ == "__main__":
    main()
