"""Shared constants for the iSTAR EEG cognitive-macrostate project.

NOTE: CHANNELS_19 order is the STANDARD 10-20 ordering inferred from the
montage image (docs/IMG_0733.JPG). The real column order inside the .mat
files must be confirmed with the professor before trusting channel selection.
"""
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "iStar_data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"    # .png plots
TABLE_DIR = OUTPUT_DIR / "tables"   # .csv data
REPORT_DIR = OUTPUT_DIR / "reports" # .txt reports / provenance

# --- acquisition (from paper; verified by inspect_data.py) ---
FS = 1000  # Hz, sampling rate stated in the paper

# --- preprocessing filter (paper: 1-30 Hz 3rd-order Butterworth IIR + 60 Hz notch) ---
FILTER_BAND = (1, 30)  # Hz, band-pass edges
FILTER_ORDER = 3       # Butterworth order
NOTCH_HZ = 60          # Hz, power-line notch
NOTCH_Q = 30           # notch quality factor

# --- artifact handling ---
# Paper rejects epochs whose amplitude exceeds +/-150 uV (eye-blink/motion).
# Data is in VOLTS (typical |value| ~1e-5 = ~10 uV), so 150 uV = 150e-6 V.
# We clip (winsorize) rather than drop whole 2 s STFT frames: dropping any
# frame touching a spike would delete ~48% of frames and empty 25 questions,
# breaking the complete-9 set; clipping caps spike power and keeps all data.
ARTIFACT_V = 150e-6

# --- channels ---
# Assumed column order of the 19-column EEG matrix (standard 10-20).
CHANNELS_19 = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
]
# The 11 task-relevant channels used in the study.
CHANNELS_11 = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
CH_IDX_11 = [CHANNELS_19.index(c) for c in CHANNELS_11]

# --- frequency bands (Hz) ---
BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}

# --- STFT / epoching (from paper) ---
WIN_SEC = 2.0     # analysis window length
STEP_SEC = 0.1    # sliding step (overlap)

# --- analysis scope ---
VISUAL_QUESTIONS = list(range(1, 10))  # Q1-Q9 (the 9 visual-textual questions)
N_CLUSTERS = 3  # Reading / Reasoning / Decision
N_RESTARTS = 10  # K-medoids restarts; keep the lowest-cost run (avoids bad local optima)
