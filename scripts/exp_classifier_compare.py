"""Experiment A: does a different Layer-2 classifier predict answer-correctness
better than the paper's logistic regression?

Keeps the exact Layer-2 protocol (nine per-question models, eq.8-9, per-question
StratifiedKFold) and only swaps the estimator. Each estimator is evaluated on
BOTH feature sets on the same 198 rows:
  - OUR Layer-1 SRI (outputs/tables/sri.csv)
  - the professor's f1/f2/f3 (the reference that reproduces the paper)

The point is the professor's "try other models" request: see whether the weak
result on our SRI is the classifier's fault (then a stronger model would lift it)
or the feature's fault (then no classifier helps). Writes
outputs/experiments/exp_classifier_compare.csv.

Run: python scripts/exp_classifier_compare.py
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from labels import load_labels
from classify import _per_question, STATES

SEED = 0

# standardise first throughout so distance/scale-sensitive models are comparable
ESTIMATORS = {
    "logistic (paper)": make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=1000)),
    "svm_rbf": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=SEED)),
    "random_forest": make_pipeline(StandardScaler(),
                                   RandomForestClassifier(n_estimators=300,
                                                          random_state=SEED)),
    "hist_grad_boost": make_pipeline(StandardScaler(),
                                     HistGradientBoostingClassifier(random_state=SEED)),
    "knn_5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

FEATURE_SETS = {"our_sri": STATES, "prof_f1f2f3": ["f1", "f2", "f3"]}


def main():
    lab = load_labels()
    sri = pd.read_csv(config.TABLE_DIR / "sri.csv")
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    assert not m.duplicated(["subject", "q"]).any(), "merge fan-out"
    print(f"merged: {len(m)} rows, {m.subject.nunique()} subjects\n")

    rows = []
    for est_name, clf in ESTIMATORS.items():
        for feat_name, cols in FEATURE_SETS.items():
            res = _per_question(m, cols, f"{est_name} | {feat_name}", clf=clf)
            rows.append({"classifier": est_name, "features": feat_name,
                         "mean_acc": res["mean_acc"], "score_r": res["score_r"],
                         "score_mae": res["score_mae"]})

    out_dir = config.OUTPUT_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = out_dir / "exp_classifier_compare.csv"
    df.to_csv(out, index=False)

    print("\n=== summary (sorted by mean_acc within feature set) ===")
    for feat_name in FEATURE_SETS:
        sub = df[df.features == feat_name].sort_values("mean_acc", ascending=False)
        print(f"\n[{feat_name}]")
        print(sub[["classifier", "mean_acc", "score_r", "score_mae"]]
              .to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
