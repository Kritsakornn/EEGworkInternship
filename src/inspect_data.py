"""Step 0: check whether the cognitive EEG is band-pass filtered, across ALL
CutEEG cognitive files the pipeline actually uses (S*/CutEEG_Cognitive,
CutEEGdata_Q1..9_*.mat) -- not just a sample. Run: python src/inspect_data.py
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


def file_shares(path, fs):
    """Per-file spectral shares (%) averaged over all channels:
    (>30Hz share, 60Hz line share). Returns None for a flat file."""
    x, _ = load_eeg(path)
    hi30, line = [], []
    for c in range(x.shape[1]):
        ch = x[:, c]
        total = band_power(ch, fs, 0.1, fs / 2)
        if total == 0:
            continue
        hi30.append(band_power(ch, fs, 30, fs / 2) / total * 100)
        line.append(band_power(ch, fs, 59, 61) / total * 100)
    if not hi30:
        return None
    return float(np.mean(hi30)), float(np.mean(line))


def main():
    fs = config.FS
    files = sorted(config.DATA_DIR.glob("S*/CutEEG_Cognitive/CutEEGdata_Q[1-9]_*.mat"))
    print(f"scanning {len(files)} cognitive files (Q1-9, all subjects) ...")

    hi30, line, flat = [], [], 0
    worst = None  # (path, >30Hz share)
    for path in files:
        res = file_shares(path, fs)
        if res is None:
            flat += 1
            continue
        h, ln = res
        hi30.append(h)
        line.append(ln)
        if worst is None or h > worst[1]:
            worst = (path.relative_to(config.DATA_DIR), h)

    hi30, line = np.array(hi30), np.array(line)
    print(f"\nfiles scanned: {len(files)}  (flat/skipped: {flat})")
    print(f">30Hz power share %: mean={hi30.mean():.2f}  median={np.median(hi30):.2f}  "
          f"min={hi30.min():.2f}  max={hi30.max():.2f}")
    print(f"60Hz line share %  : mean={line.mean():.3f}  max={line.max():.3f}")
    print(f"files with >30Hz share >1%: {(hi30 > 1).sum()} / {len(hi30)}")
    print(f"files with >30Hz share >5%: {(hi30 > 5).sum()} / {len(hi30)}")
    print(f"worst file: {worst[0]}  (>30Hz={worst[1]:.1f}%)")

    print("\nInterpretation:")
    print("  - 60Hz line share ~0% across all files -> line noise was notched out.")
    print("  - >30Hz share NOT near 0% -> data is NOT hard band-pass limited to")
    print("    1-30Hz; high-freq content (likely EMG) remains in many files.")
    print("  - RSP features are unaffected: they sum only theta/alpha/beta (4-30Hz)")
    print("    and normalize within them, so >30Hz power is excluded by construction.")


if __name__ == "__main__":
    main()
