"""Deliverable #1: EEG spectral power per (subject, question, cognitive state).

For every (subject, question, state) we average the relative spectral power
(RSP) across the 11 channels, giving theta/alpha/beta values. Per-row
theta+alpha+beta sums to 1 (each frame's per-channel bands already sum to 1).

Run: python src/spectral_power.py
"""
import pandas as pd

import config
from macrostates import cluster_frames, STATES

# 33-dim vector layout: channel i -> [i*3+0=theta, i*3+1=alpha, i*3+2=beta]
BAND_OFFSET = {"theta": 0, "alpha": 1, "beta": 2}


def main():
    X, groups, labels, names = cluster_frames()

    df = pd.DataFrame(groups, columns=["subject", "q"])
    df["state"] = [names[l] for l in labels]
    # mean RSP across the 11 channels, per band
    for band, off in BAND_OFFSET.items():
        cols = [c * 3 + off for c in range(len(config.CHANNELS_11))]
        df[band] = X[:, cols].mean(axis=1)

    agg = (df.groupby(["subject", "q", "state"])
             .agg(n_frames=("theta", "size"),
                  theta=("theta", "mean"),
                  alpha=("alpha", "mean"),
                  beta=("beta", "mean"))
             .reset_index())

    # order states meaningfully
    agg["state"] = pd.Categorical(agg["state"], categories=STATES, ordered=True)
    agg = agg.sort_values(["subject", "q", "state"]).reset_index(drop=True)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "spectral_power.csv"
    agg.to_csv(out, index=False)

    print(agg.head(12).to_string(index=False))
    chk = agg[["theta", "alpha", "beta"]].sum(axis=1)
    print(f"\nrows={len(agg)}  band-sum: min={chk.min():.4f} max={chk.max():.4f} "
          f"(expect 1.0)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
