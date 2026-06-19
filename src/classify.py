"""Layer 2: per-question performance prediction (paper eqs. 7-9).

Professor deliverables #2 + #3. The paper builds NINE item-specific logistic
regression models (one per question, eq.8), NOT a single pooled model, because
the nine items differ in difficulty and cognitive demand. Each subject appears
once per question, so a within-question k-fold split is already subject-
independent (no subject spans train and test). Per-question out-of-fold accuracy
reproduces the paper's Table 2 (0.725-0.875). The nine binary predictions are
then summed per subject into a predicted total score and correlated with the
actual score (eq.9; the thesis reports test MAE ~1.0, Pearson r ~0.74).

Two feature sets compared on the same rows:
  - OUR SRI from outputs/tables/sri.csv  -> tests our Layer-1 end-to-end
  - the professor's f1/f2/f3 from the labels -> reference baseline

Run: python src/classify.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr

import config
from labels import load_labels

STATES = ["Reading", "Reasoning", "Decision"]
N_SPLITS = 5
SEED = 0


def _per_question(m, cols, name, clf=None):
    """Nine item-specific logistic models (eq.8) + score-level aggregate (eq.9).

    Returns a result dict; prints per-question accuracy and the subject-level
    predicted-vs-actual total-score correlation. `clf` overrides the estimator
    (default = standardised logistic regression, the paper's model); each fold
    clones it, so one instance is reused across all nine questions.
    """
    if clf is None:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    accs, preds = {}, pd.Series(index=m.index, dtype=float)
    for q, g in m.groupby("q"):
        y = g.label.values
        # within a question each subject has one row, so any fold split is
        # subject-independent; k clamps to the minority-class size for tiny Qs
        k = min(N_SPLITS, int(np.bincount(y).min()))
        if k < 2:  # a near single-class question can't be cross-validated
            raise ValueError(f"Q{q}: minority class has <2 subjects; cannot "
                             f"cross-validate (label balance {np.bincount(y).tolist()})")
        p = cross_val_predict(clf, g[cols].values, y,
                              cv=StratifiedKFold(k, shuffle=True, random_state=SEED))
        accs[q] = accuracy_score(y, p)
        preds.loc[g.index] = p

    # eq.9: sum the nine per-question predictions per subject -> predicted score
    by_subj = m.assign(_pred=preds).groupby("subject").agg(
        pred_score=("_pred", "sum"), true_score=("label", "sum"))
    if by_subj.pred_score.std() == 0 or by_subj.true_score.std() == 0:
        r = float("nan")  # constant predictions -> correlation undefined
    else:
        r = pearsonr(by_subj.pred_score, by_subj.true_score)[0]
    mae = mean_absolute_error(by_subj.true_score, by_subj.pred_score)
    mean_acc = float(np.mean(list(accs.values())))

    print(f"\n[{name}]")
    print("  per-Q accuracy:",
          " ".join(f"Q{q}={accs[q]:.3f}" for q in sorted(accs)))
    print(f"  mean per-Q accuracy = {mean_acc:.3f}   (paper Table 2: 0.725-0.875)")
    print(f"  score-level: Pearson r = {r:.3f}  MAE = {mae:.3f}   "
          f"(thesis test: r~0.74, MAE~1.0)")
    return {"name": name, "mean_acc": round(mean_acc, 3),
            "score_r": round(r, 3), "score_mae": round(mae, 3)}


def main():
    lab = load_labels()
    sri = pd.read_csv(config.TABLE_DIR / "sri.csv")  # subject,q,Reading,Reasoning,Decision
    m = sri.merge(lab, on=["subject", "q"], how="inner")
    # inner merge on a unique key must not fan out (would inflate n, bias folds)
    assert not m.duplicated(["subject", "q"]).any(), \
        "merge fan-out: duplicate (subject, q) keys"
    print(f"merged: {len(m)} rows, {m.subject.nunique()} subjects, "
          f"{m.q.nunique()} questions")

    results = [
        _per_question(m, STATES, "OUR SRI (Reading/Reasoning/Decision)"),
        _per_question(m, ["f1", "f2", "f3"], "Professor f1/f2/f3"),
    ]

    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = config.TABLE_DIR / "layer2_classification.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
