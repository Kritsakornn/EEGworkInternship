"""Steps 1-3: load cognitive EEG for a question, select 11 channels, and
compute relative spectral power (RSP) features via STFT.

Each STFT time frame (2 s window, 0.1 s step) becomes one 33-dim feature
vector: 11 channels x {theta, alpha, beta}, ordered as in the paper eq. (3):
[RSP_theta_Fp1, RSP_alpha_Fp1, RSP_beta_Fp1, ..., RSP_beta_P4].

The cognitive EEG is already band-pass filtered (~1-30 Hz, verified by
inspect_data.py), so no extra filtering is applied here.

Run: python src/features.py
"""
import numpy as np
import scipy.io as sio
from scipy.signal import stft

import config


def find_cognitive_files(subject, q):
    """Return sorted cognitive segment files for one question (may be >1)."""
    folder = config.DATA_DIR / subject / "CutEEG_Cognitive"
    return sorted(folder.glob(f"CutEEGdata_Q{q}_*.mat"))


def load_question_eeg(subject, q):
    """Load + concatenate all cognitive segments of a question -> (n_samples, 19)."""
    parts = []
    for f in find_cognitive_files(subject, q):
        m = sio.loadmat(f)
        key = [k for k in m if not k.startswith("__")][0]
        parts.append(np.asarray(m[key], dtype=float))
    if not parts:
        raise FileNotFoundError(f"no cognitive files for {subject} Q{q}")
    eeg = np.concatenate(parts, axis=0)
    # clip eye-blink/motion spikes to +/-150 uV (paper artifact threshold) so a
    # 30 mV transient cannot dominate a frame's spectral power; keeps all frames
    return np.clip(eeg, -config.ARTIFACT_V, config.ARTIFACT_V)


def _rsp(eeg, ch_idx):
    """RSP features for the given channel indices -> (n_frames, 3*len(ch_idx))."""
    sel = eeg[:, ch_idx]
    # 2 s window, 0.1 s step; clamp to signal length for short segments
    nperseg = min(int(config.WIN_SEC * config.FS), sel.shape[0])
    noverlap = max(0, nperseg - int(config.STEP_SEC * config.FS))

    per_channel = []
    for ch in range(sel.shape[1]):
        f, _, zxx = stft(sel[:, ch], fs=config.FS,
                         nperseg=nperseg, noverlap=noverlap, boundary=None)
        power = np.abs(zxx) ** 2  # (freq, frame)
        band_powers = []
        for lo, hi in config.BANDS.values():
            mask = (f >= lo) & (f < hi)
            band_powers.append(power[mask].sum(axis=0))  # (frame,)
        bands = np.stack(band_powers, axis=1)            # (frame, 3) theta,alpha,beta
        total = bands.sum(axis=1, keepdims=True)
        rsp = np.divide(bands, total, out=np.zeros_like(bands), where=total > 0)
        per_channel.append(rsp)                          # (frame, 3)

    return np.concatenate(per_channel, axis=1)           # (frame, 3*n_ch)


def rsp_features(eeg19):
    """(n_samples, 19) -> (n_frames, 33): RSP for the 11 task channels."""
    return _rsp(eeg19, config.CH_IDX_11)


def rsp_features_allch(eeg19):
    """(n_samples, 19) -> (n_frames, 57): RSP for all 19 channels (for topomaps)."""
    return _rsp(eeg19, list(range(eeg19.shape[1])))


def main():
    subject, q = "S1", 2
    eeg = load_question_eeg(subject, q)
    print(f"{subject} Q{q}: loaded EEG {eeg.shape} (samples x channels)")

    feats = rsp_features(eeg)
    print(f"features: {feats.shape} (frames x 33)")
    print(f"  value range: min={feats.min():.3f} max={feats.max():.3f}")
    # each frame's 3 bands per channel must sum to 1
    sums = feats.reshape(feats.shape[0], 11, 3).sum(axis=2)
    print(f"  per-channel RSP sum: min={sums.min():.4f} max={sums.max():.4f} "
          f"(expect ~1.0)")


if __name__ == "__main__":
    main()
