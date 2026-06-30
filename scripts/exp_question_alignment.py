"""Phase-3 test of the question-index alignment hypothesis (project audit).

HYPOTHESIS: labels.load_labels keys each label file by its FILENAME index
(01..09total.xlsx -> q=1..9) and ignores the internal `question` column, whose
true IDs are {1,2,3,6,8,11,15,16,17}. EEG cognitive cuts are numbered
{1..9,13,18}. The committed Layer-2 merges SRI(EEG q) with label(filename idx q),
so only q in {1,2,3} are guaranteed aligned. If the hypothesis is right,
re-keying labels by the TRUE `question` id should improve Layer-2 accuracy on the
questions whose pairing changes (EEG q=6, q=8), while q=1,2,3 stay fixed (control).

Minimal change: ONLY the label merge key changes. SRI, features, clustering and
the classifier are identical to the committed pipeline (classify._per_question).

Run: python scripts/exp_question_alignment.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from classify import _per_question
from labels import load_labels  # committed: filename-index keyed

LABEL_DIR = config.DATA_DIR / "label_answer_data"
STATES = ["Reading", "Reasoning", "Decision"]


def load_labels_true_q():
    """labels.load_labels, but q = internal `question` id (true), not filename idx."""
    frames = []
    for f in sorted(LABEL_DIR.glob("[0-9][0-9]total.xlsx")):
        d = pd.read_excel(f)[["subject", "question", "f1", "f2", "f3", "label"]]
        d = d[d.subject.astype(str).str.match(r"^s\d+$", case=False, na=False)]
        d = d[d.label.isin([0, 1])]
        d["q"] = d["question"].astype(int)
        d["subject"] = d.subject.str.upper()
        frames.append(d)
    lab = pd.concat(frames, ignore_index=True)
    lab["label"] = lab.label.astype(int)
    assert not lab.duplicated(["subject", "q"]).any(), "dup (subject,q) in true-q labels"
    return lab[["subject", "q", "f1", "f2", "f3", "label"]]


def show_mapping():
    print("filename-index  ->  true question id (from the `question` column)")
    for f in sorted(LABEL_DIR.glob("[0-9][0-9]total.xlsx")):
        d = pd.read_excel(f)
        real = d[d.subject.astype(str).str.match(r"^s\d+$", case=False, na=False)]
        print(f"  {f.name}  filename_q={int(f.name[:2])}  ->  question={int(real['question'].dropna().iloc[0])}")


def main():
    show_mapping()
    sri = pd.read_csv(config.TABLE_DIR / "sri.csv")
    lab_file = load_labels()
    lab_true = load_labels_true_q()
    print("\nEEG SRI q       :", sorted(sri.q.unique()))
    print("label filename q:", sorted(lab_file.q.unique()))
    print("label TRUE q    :", sorted(lab_true.q.unique()))

    overlap = sorted(set(sri.q) & set(lab_true.q))
    print("alignable EEG questions (EEG SRI ∩ TRUE labels):", overlap)
    sri_ov = sri[sri.q.isin(overlap)]

    m_cur = sri_ov.merge(lab_file, on=["subject", "q"], how="inner")
    m_aln = sri_ov.merge(lab_true, on=["subject", "q"], how="inner")
    print(f"\nrows  current={len(m_cur)}  aligned={len(m_aln)}  "
          f"(q1-3 identical; q6,q8 re-paired)")

    print("\n================ CURRENT keying (committed) ================")
    _per_question(m_cur, STATES, f"SRI current keying  q={overlap}")
    print("\n================ TRUE-question keying (the fix) ============")
    _per_question(m_aln, STATES, f"SRI true-q keying   q={overlap}")
    print("\n--------- reference upper bound on the same aligned rows ---------")
    _per_question(m_aln, ["f1", "f2", "f3"], "Professor f1/f2/f3 (true-q rows)")


if __name__ == "__main__":
    main()
