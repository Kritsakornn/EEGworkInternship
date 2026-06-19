"""Layer-2 figure: per-question classification accuracy with std (paper Table 2).

The paper reports Layer-2 as per-question accuracy of the nine item-specific
logistic models (eq.8), summarised as Mean Accuracy +/- SD (0.829 +/- 0.046).
This reproduces that as a grouped bar chart (Q1-Q9), one group per feature set:

  - OUR SRI from outputs/tables/sri.csv
  - the professor's f1/f2/f3 from the labels

Error bars are the std across CV folds within each question (per-question
uncertainty); the title annotates each set's Mean +/- SD across the nine
questions (the paper's reported form). Same clf / fold protocol as classify.py.

Run: python src/plot_layer2.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import config
from classify import STATES, N_SPLITS, SEED  # reuse the exact Layer-2 protocol
from labels import load_labels

FEATURE_SETS = [
    ("Our SRI", STATES),
    ("Professor f1/f2/f3", ["f1", "f2", "f3"]),
]


def per_question_accuracy(m, cols):
    """-> (questions, mean_acc, std_acc) using the same fold protocol as classify.py.

    cross_val_score gives the per-fold accuracies, so std across folds is the
    per-question error bar; classify.py's cross_val_predict only yields a single
    pooled point estimate with no spread.
    """
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    qs, means, stds = [], [], []
    for q, g in m.groupby("q"):
        y = g.label.values
        k = min(N_SPLITS, int(np.bincount(y).min()))  # clamp to minority class
        scores = cross_val_score(clf, g[cols].values, y, scoring="accuracy",
                                 cv=StratifiedKFold(k, shuffle=True, random_state=SEED))
        qs.append(q)
        means.append(scores.mean())
        stds.append(scores.std())
    return np.array(qs), np.array(means), np.array(stds)


def main():
    lab = load_labels()
    sri = pd.read_csv(config.TABLE_DIR / "sri.csv")
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"

    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5))
    rows = []
    for i, (name, cols) in enumerate(FEATURE_SETS):
        qs, mean, std = per_question_accuracy(m, cols)
        x = np.arange(len(qs)) + (i - 0.5) * width
        overall_m, overall_sd = mean.mean(), mean.std()
        ax.bar(x, mean, width, yerr=std, capsize=3,
               label=f"{name}  (mean {overall_m:.3f} ± {overall_sd:.3f})")
        for q, mu, sd in zip(qs, mean, std):
            rows.append({"feature_set": name, "q": int(q),
                         "acc_mean": round(mu, 3), "acc_std": round(sd, 3)})

    ax.axhline(0.5, ls="--", lw=0.8, color="grey", label="chance (0.5)")
    ax.set_xticks(np.arange(len(qs)))
    ax.set_xticklabels([f"Q{q}" for q in qs])
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Layer-2 per-question answer-correctness accuracy "
                 "(mean ± SD over CV folds)\npaper Table 2: SRI-feature "
                 "mean 0.829 ± 0.046")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIG_DIR / "layer2_accuracy.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved -> {out}")

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv = config.TABLE_DIR / "layer2_per_question.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"saved -> {csv}")


if __name__ == "__main__":
    main()
