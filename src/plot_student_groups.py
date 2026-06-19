"""Paper-style figure: cognitive-state proportion by student achievement group.

Reproduces the paper's "Proportion Distribution of Three Cognitive States
Across Questions": a 3x3 facet (Q1-Q9), each panel showing the mean SRI
proportion (%) of the three states for two student groups, split by total
answer-correctness score (sum of the 9 labels):

  - High-Achieving (Top Group)    : score >= 6  (above the median of 5)
  - Average-Achieving (Middle Grp): score <= 5

The insight is to SEPARATE the two student groups by how they distribute their
cognitive states, NOT to compare feature sources. Proportions are our Layer-1
SRI from outputs/tables/sri.csv.

Run: python src/plot_student_groups.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import config
from macrostates import STATES  # Reading, Reasoning, Decision
from labels import load_labels

HIGH_COLOR = "#FFC000"  # gold  -> Top group
AVG_COLOR = "#5B9BD5"   # blue  -> Middle group
SCORE_CUT = 6           # High = total score >= 6 (median split; median = 5)
STATE_TICKS = ["1 (Reading)", "2 (Reasoning)", "3 (Decision)"]


def main():
    lab = load_labels()
    sri = pd.read_csv(config.TABLE_DIR / "sri.csv")
    m = sri.merge(lab[["subject", "q", "label"]], on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"

    # total answer-correctness score per subject -> achievement group
    score = m.groupby("subject").label.sum()
    high = set(score[score >= SCORE_CUT].index)
    m["group"] = np.where(m.subject.isin(high), "High", "Average")
    m[STATES] = m[STATES] * 100  # proportions -> percent
    n_high, n_avg = len(high), score.size - len(high)
    print(f"High (score>={SCORE_CUT}): {n_high} subj   "
          f"Average (score<={SCORE_CUT - 1}): {n_avg} subj")

    x = np.arange(len(STATES))
    width = 0.4
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
    for ax, q in zip(axes.flat, config.VISUAL_QUESTIONS):
        sub = m[m.q == q]
        hi = sub[sub.group == "High"][STATES]
        av = sub[sub.group == "Average"][STATES]
        # error bars = std across subjects within the group (between-subject spread)
        ax.bar(x - width / 2, hi.mean().values, width, yerr=hi.std().values,
               capsize=2, ecolor="grey", color=HIGH_COLOR)
        ax.bar(x + width / 2, av.mean().values, width, yerr=av.std().values,
               capsize=2, ecolor="grey", color=AVG_COLOR)
        ax.set_title(f"Q{q}")
        ax.set_xticks(x)
        ax.set_xticklabels(STATE_TICKS, fontsize=8)
        ax.set_xlabel("State", fontsize=8)
        ax.set_ylim(0, 60)
        ax.grid(axis="y", ls="--", lw=0.5, alpha=0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Percentage (%)")

    fig.suptitle("Proportion Distribution of Three Cognitive States Across Questions",
                 fontsize=14)
    handles = [
        Patch(color=HIGH_COLOR, label=f"High-Achieving Students (Top Group, n={n_high})"),
        Patch(color=AVG_COLOR, label=f"Average-Achieving Students (Middle Group, n={n_avg})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.045))
    fig.text(0.5, 0.01,
             "Note:  1 = Reading state;  2 = Reasoning state;  3 = Decision state.",
             ha="center", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="grey", ls="--"))
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIG_DIR / "sri_by_group.png"
    fig.savefig(out, dpi=120)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
