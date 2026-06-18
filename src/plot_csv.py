"""Quick-look figures from the output CSVs.

- sri.png: mean Reading/Reasoning/Decision proportion per question (stacked).
- band_power.png: mean theta/alpha/beta RSP per cognitive state (grouped).

Run: python src/plot_csv.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from macrostates import STATES

BANDS = list(config.BANDS)  # ["theta", "alpha", "beta"]


def plot_sri():
    df = pd.read_csv(config.TABLE_DIR / "sri.csv")
    mean = df.groupby("q")[STATES].mean()  # (9, 3), rows sum to ~1

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = 0
    for state in STATES:
        ax.bar(mean.index, mean[state], bottom=bottom, label=state)
        bottom = bottom + mean[state].values
    ax.set_xlabel("Question")
    ax.set_ylabel("Mean SRI proportion")
    ax.set_title("Cognitive-state proportion per question (mean over subjects)")
    ax.set_xticks(mean.index)
    ax.legend(title="State")
    return fig


def plot_band_power():
    df = pd.read_csv(config.TABLE_DIR / "spectral_power.csv")
    g = df.groupby("state")[BANDS]
    mean, std = g.mean().reindex(STATES), g.std().reindex(STATES)

    x = range(len(STATES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, band in enumerate(BANDS):
        ax.bar([xi + (i - 1) * width for xi in x], mean[band], width,
               yerr=std[band], capsize=3, label=band)
    ax.set_xticks(list(x))
    ax.set_xticklabels(STATES)
    ax.set_ylabel("Mean RSP")
    ax.set_title("Band power per cognitive state (mean over subjects/questions)")
    ax.legend(title="Band")
    return fig


def plot_band_power_by_question():
    """Grouped band-power-per-state bars, faceted per question (Q1-Q9)."""
    df = pd.read_csv(config.TABLE_DIR / "spectral_power.csv")
    questions = config.VISUAL_QUESTIONS
    x = range(len(STATES))
    width = 0.25

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharey=True)
    for ax, q in zip(axes.flat, questions):
        mean = df[df["q"] == q].groupby("state")[BANDS].mean().reindex(STATES)
        for i, band in enumerate(BANDS):
            ax.bar([xi + (i - 1) * width for xi in x], mean[band], width,
                   label=band)
        ax.set_xticks(list(x))
        ax.set_xticklabels(STATES, fontsize=8)
        ax.set_title(f"Q{q}")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean RSP")
    axes[0, 0].legend(title="Band", fontsize=8)
    fig.suptitle("Band power per cognitive state, by question (mean over subjects)")
    fig.tight_layout()
    return fig


def plot_sri_dist():
    """Boxplots: spread of each state's SRI across all 243 subject-questions."""
    df = pd.read_csv(config.TABLE_DIR / "sri.csv")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([df[s] for s in STATES], tick_labels=STATES, showmeans=True)
    ax.set_ylabel("SRI proportion")
    ax.set_title("Distribution of SRI per state (each point = one subject-question)")
    return fig


def plot_sri_heatmap():
    """Heatmap of Reasoning SRI for every subject x question (between-subject view)."""
    df = pd.read_csv(config.TABLE_DIR / "sri.csv")
    df = df.sort_values("subject", key=lambda s: s.str[1:].astype(int))
    pivot = df.pivot(index="subject", columns="q", values="Reasoning")

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Question")
    ax.set_title("Reasoning SRI per subject x question")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Reasoning proportion")
    return fig


def plot_band_dist():
    """Boxplots: spread of theta/alpha/beta RSP per state (state separability)."""
    df = pd.read_csv(config.TABLE_DIR / "spectral_power.csv")
    fig, axes = plt.subplots(1, len(BANDS), figsize=(11, 4), sharey=True)
    for ax, band in zip(axes, BANDS):
        ax.boxplot([df[df["state"] == s][band] for s in STATES],
                   tick_labels=STATES, showmeans=True)
        ax.set_title(band)
    axes[0].set_ylabel("RSP")
    fig.suptitle("Band-power distribution per state")
    fig.tight_layout()
    return fig


def main():
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = [("sri", plot_sri()), ("band_power", plot_band_power()),
            ("band_power_by_question", plot_band_power_by_question()),
            ("sri_dist", plot_sri_dist()), ("sri_heatmap", plot_sri_heatmap()),
            ("band_dist", plot_band_dist())]
    for name, fig in figs:
        out = config.FIG_DIR / f"{name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
