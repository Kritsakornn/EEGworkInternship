"""Task B: validate the state mapping with per-state band-power topomaps.

The paper normalizes by a 3-min resting baseline (change-ratio, eq.10), but no
resting EEG exists in this dataset, so this plots mean RSP per state directly.
It validates the spatial pattern (e.g. is frontal theta highest in Reasoning?),
not the exact paper figure.

Run: python src/topomap.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

import config
from features import load_question_eeg, rsp_features_allch
from macrostates import cluster_frames, complete_subjects, STATES

BANDS = list(config.BANDS)  # ["theta", "alpha", "beta"]
# old 10-20 names -> modern names known to MNE's standard_1020 montage
RENAME = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
FRONTAL = ["Fp1", "Fp2", "F3", "Fz", "F4"]


def build_allch_matrix(subjects):
    feats = []
    for s in subjects:
        for q in config.VISUAL_QUESTIONS:
            feats.append(rsp_features_allch(load_question_eeg(s, q)))
    return np.concatenate(feats)


def main():
    _, _, labels, names = cluster_frames()
    subjects = complete_subjects()
    A = build_allch_matrix(subjects)  # (N, 57) = 19 ch x 3 bands
    assert len(A) == len(labels), (len(A), len(labels))
    state_of = np.array([names[l] for l in labels])

    ch_names = [RENAME.get(c, c) for c in config.CHANNELS_19]
    info = mne.create_info(ch_names, config.FS, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))

    fig, axes = plt.subplots(len(STATES), len(BANDS), figsize=(9, 9))
    # mean RSP map per (state, band); collect first so each band shares one
    # color scale across the 3 states (otherwise panels auto-scale separately
    # and look identically flat).
    maps = {}
    for r, state in enumerate(STATES):
        mask = state_of == state
        for c, band in enumerate(BANDS):
            cols = [ch * 3 + c for ch in range(19)]
            maps[(r, c)] = A[mask][:, cols].mean(axis=0)  # (19,)

    for c, band in enumerate(BANDS):
        col_vals = np.stack([maps[(r, c)] for r in range(len(STATES))])
        vlim = (col_vals.min(), col_vals.max())
        for r, state in enumerate(STATES):
            im, _ = mne.viz.plot_topomap(
                maps[(r, c)], info, axes=axes[r, c],
                cmap="viridis", vlim=vlim, show=False,
            )
            axes[r, c].set_title(f"{state} {band}")
        fig.colorbar(im, ax=axes[:, c].tolist(), shrink=0.6, label=f"{band} RSP")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "topomaps.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")

    front_idx = [config.CHANNELS_19.index(c) * 3 + 0 for c in FRONTAL]  # theta
    print("frontal theta mean per state (expect Reasoning > Decision > Reading):")
    for state in STATES:
        print(f"  {state:9s}: {A[state_of == state][:, front_idx].mean():.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
