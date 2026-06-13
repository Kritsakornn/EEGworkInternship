"""Step 0: verify sampling rate assumption & whether the cognitive EEG is
already band-pass filtered. Run: python src/inspect_data.py
"""
import numpy as np
import scipy.io as sio
from scipy.signal import welch

import config


def load_eeg(path):
    m = sio.loadmat(path)
    key = [k for k in m if not k.startswith("__")][0]
    return np.asarray(m[key], dtype=float), key


def band_power(x, fs, fmin, fmax):
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    mask = (f >= fmin) & (f < fmax)
    return float(np.trapz(pxx[mask], f[mask]))


def main():
    fs = config.FS
    samples = {
        "cognitive_Q2_2": config.DATA_DIR / "S1/CutEEG_Cognitive/CutEEGdata_Q2_2.mat",
        "full_Q2": config.DATA_DIR / "S1/CutEEG_Cognitive/EEG_All_Questions/iStarCutEEGdata_Q2.mat",
    }
    for name, path in samples.items():
        x, key = load_eeg(path)
        print(f"\n=== {name} ({key}) ===")
        print(f"  shape={x.shape}  duration@{fs}Hz = {x.shape[0]/fs:.1f}s")
        print(f"  value range: min={x.min():.2f}  max={x.max():.2f}  "
              f"mean={x.mean():.3f}  std={x.std():.2f}")
        print(f"  NaN={np.isnan(x).sum()}  Inf={np.isinf(x).sum()}")

        ch = x[:, 0]  # first channel for spectrum check
        total = band_power(ch, fs, 0.1, fs / 2)
        if total == 0:
            print("  (flat signal, skip spectrum)")
            continue
        checks = {
            "<1Hz (drift)": (0.1, 1),
            "1-30Hz (kept band)": (1, 30),
            "30-50Hz": (30, 50),
            "60Hz (line)": (59, 61),
            ">50Hz": (50, fs / 2),
        }
        print("  spectral power share (ch0):")
        for label, (lo, hi) in checks.items():
            print(f"    {label:20s}: {band_power(ch, fs, lo, hi)/total*100:5.1f}%")

    print("\nInterpretation:")
    print("  - If >30Hz share is near 0% -> already low-pass/band-pass filtered.")
    print("  - If 60Hz spike present     -> raw (line noise not removed).")


if __name__ == "__main__":
    main()
