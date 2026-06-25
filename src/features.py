"""Steps 1-3: load cognitive EEG for a question, select 11 channels, and
compute relative spectral power (RSP) features via STFT.

Each STFT time frame (2 s window, 0.1 s step) becomes one 33-dim feature
vector: 11 channels x {theta, alpha, beta}, ordered as in the paper eq. (3):
[RSP_theta_Fp1, RSP_alpha_Fp1, RSP_beta_Fp1, ..., RSP_beta_P4].

Preprocessing matches the paper: a 1-30 Hz third-order Butterworth IIR
band-pass plus a 60 Hz notch (config.FILTER_*), applied per channel in
load_question_eeg. STFT frames whose 2 s window exceeds +/-150 uV are then
rejected (dropped) in _rsp, as in the paper. Most CutEEG files arrived
already filtered (line noise removed, ~89% power in 1-30 Hz), so re-applying
is near-idempotent on those; it cleans the minority that were under-filtered
(some retained >100 Hz content a 1-30 band-pass cannot leave -- see
inspect_data.py). A 3rd-order Butterworth is a gentle roll-off, not a brick
wall, so ~10% residual >30 Hz is expected and harmless: RSP sums only
theta/alpha/beta (4-30 Hz) and normalizes within them.

Run: python src/features.py
"""
from matplotlib import backend_bases
from cProfile import label
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, iirnotch, stft, welch
import matplotlib.pyplot as plt

import config


def _bandpass_notch(eeg):
    """Paper preprocessing: 1-30 Hz 3rd-order Butterworth IIR band-pass + 60 Hz
    notch, applied per channel (zero-phase filtfilt). Near-idempotent on the
    already-filtered files; cleans the minority that arrived under-filtered."""
    lo, hi = config.FILTER_BAND
    b, a = butter(config.FILTER_ORDER, [lo, hi], btype="band", fs=config.FS)
    bn, an = iirnotch(config.NOTCH_HZ, config.NOTCH_Q, fs=config.FS)
    eeg = filtfilt(b, a, eeg, axis=0)
    """
    # ------- FOR TESTING AND PLOTTING(Filter EEG) ------
    plt.figure(figsize=(14,6))
    plt.plot(filtfilt(bn, an, eeg, axis=0))
    plt.title(f"EEG after filter")
    plt.axhline(y=150e-6, color='r', linestyle='--', linewidth=1.5, label='Upper Threshold (+150 $\mu$V)')
    plt.axhline(y=-150e-6, color='r', linestyle='--', linewidth=1.5, label='Lower Threshold (-150 $\mu$V)')
    plt.show()
    """
    return filtfilt(bn, an, eeg, axis=0)


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
    #eeg = np.concatenate(parts, axis=0)
    eeg = parts[-1]


    """
    # ------- FOR TESTING AND PLOTTING(Spectrum) ------
    eeg_filtered = _bandpass_notch(eeg)
    freqs_raw, psd_raw = welch(eeg[:,0], fs=1000, nperseg=int(2 * 1000))
    freqs_filt, psd_filt = welch(eeg_filtered[:,0], fs=1000, nperseg=int(2 * 1000))

    plt.figure(figsize=(12, 6))

    plt.semilogy(
        freqs_raw, psd_raw, label="Raw EEG Spectrum", color="red", alpha=0.6
    )
    plt.semilogy(
        freqs_filt, psd_filt, label="Filtered EEG Spectrum", color="blue", lw=1.5
    )

    plt.xlim(0, 80)

    
    # Draw vertical guide lines for filter boundaries
    plt.axvline(1, color="gray", linestyle="--", alpha=0.5, label="1 Hz / 30 Hz Cutoffs")
    plt.axvline(30, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(60, color="purple", linestyle=":", alpha=0.7, label="60 Hz Notch")

    plt.title(f"Frequency Spectrum Verification")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2 / Hz)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()
    """


    
    # ------- FOR TESTING AND PLOTTING (Raw EEG) ------
    """
    plt.figure(figsize=(14,6))
    plt.plot(eeg)
    plt.title(f"{subject} Q{q}: EEG before filter")
    plt.axvline(x=len(eeg[:,5])/2, color='b', linestyle='-', label='Half')
    plt.axhline(y=150e-6, color='r', linestyle='--', linewidth=1.5, label='Upper Threshold (+150 $\mu$V)')
    plt.axhline(y=-150e-6, color='r', linestyle='--', linewidth=1.5, label='Lower Threshold (-150 $\mu$V)')
    plt.legend()
    plt.show()
    """

    return _bandpass_notch(eeg)  # paper 1-30 Hz band-pass + 60 Hz notch


def _rsp(eeg, ch_idx, mask_idx=None):
    """RSP features for the given channel indices -> (n_frames, 3*len(ch_idx)).

    mask_idx selects which channels define the artifact-rejection mask; defaults
    to ch_idx. Pass the task channels here when featurising all 19 channels so
    the surviving frames match the 11-channel clustering frame-for-frame."""
    sel = eeg[:, ch_idx]
    mask_sel = sel if mask_idx is None else eeg[:, mask_idx]
    # 2 s window, 0.1 s step; clamp to signal length for short segments
    nperseg = min(int(config.WIN_SEC * config.FS), sel.shape[0])
    noverlap = max(0, nperseg - int(config.STEP_SEC * config.FS))

    # paper artifact rejection: drop any STFT frame whose 2 s window contains a
    # sample exceeding +/-150 uV in any analysed channel (eye-blink/motion).
    # Replaces clipping: a clipped spike still injects broadband power, so the
    # contaminated frame is removed rather than capped.
    bad_sample = (np.abs(mask_sel) > config.ARTIFACT_V).any(axis=1)  # (n_samples,)
    hop = nperseg - noverlap

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

    feats = np.concatenate(per_channel, axis=1)          # (frame, 3*n_ch)
    # drop frames whose 2 s window touched an artifact spike (paper rejection)
    keep = np.array([not bad_sample[i * hop:i * hop + nperseg].any()
                     for i in range(feats.shape[0])], dtype=bool)
    return feats[keep]


def rsp_features(eeg19):
    """(n_samples, 19) -> (n_frames, 33): RSP for the 11 task channels."""
    return _rsp(eeg19, config.CH_IDX_11)


def rsp_features_allch(eeg19):
    """(n_samples, 19) -> (n_frames, 57): RSP for all 19 channels (for topomaps).

    Rejects frames using the 11 task channels (CH_IDX_11), so the surviving
    rows line up 1:1 with rsp_features() and the cluster labels built from it."""
    return _rsp(eeg19, list(range(eeg19.shape[1])), mask_idx=config.CH_IDX_11)


def main():
    subject, q = "S13", 6
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
