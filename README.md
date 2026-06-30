# iSTAR EEG — Cognitive Macrostate

Python reproduction of the IEEE/thesis study *"EEG-based cognitive macrostate
estimation of students' problem-solving process"* (iSTAR). The original work is
in `docs/` (IEEE paper + Chinese thesis `.docx`); this repo reimplements its
pipeline in Python so the analysis is free, reproducible, and reads the `.mat`
data directly with `scipy`.

**Goal.** Turn continuous EEG recorded while students solve visual–textual
science questions into a sequence of three *cognitive macrostates*
(**Reading / Reasoning / Decision**), summarise each question by its **State
Ratio Index (SRI)** = the time-proportion spent in each state, then use the SRI
to predict whether the student answered correctly.

---

## What the professor asked for (deliverables)

1. EEG spectral power per **subject × question × cognitive state** → `outputs/tables/spectral_power.csv`
2. **Classification** of correct-vs-incorrect answering → `outputs/tables/layer2_classification.csv`
3. **Cross-validation** of that classifier
4. Analyse **only subjects with complete 9-question data**

Full requirement text: `docs/RequirementOfProfessor.md`. Thai summary of the
source paper + method: `docs/สรุปงานวิจัย_ภาษาไทย.md`.

---

## Pipeline (two layers)

```
Layer 1  (unsupervised — no labels needed)
  load cognitive EEG  .mat (samples × 19)        features.py
    → dedupe redundant segment files
    → 1–30 Hz Butterworth(3) band-pass + 60 Hz notch
    → select 11 task channels
    → STFT (2 s window, 0.1 s step)
    → drop any frame whose 2 s window exceeds ±150 µV   (artifact rejection)
    → Relative Spectral Power θ/α/β  → 33-dim vector per frame
  K-medoids (K=3, 10 restarts, lowest cost)            macrostates.py
    → map clusters to Reading/Reasoning/Decision by frontal-theta order
    → SRI per (subject, question)                  → outputs/tables/sri.csv
  per-state mean band power                            spectral_power.py
                                                  → outputs/tables/spectral_power.csv

Layer 2  (supervised — uses the professor's answer labels)
  9 per-question logistic-regression models (one per Q)   classify.py
    → 5-fold stratified CV (subject-independent: 1 row / subject / question)
    → sum the 9 predictions per subject → predicted score, correlate with actual
    → compares OUR SRI vs the professor's f1/f2/f3   → outputs/tables/layer2_classification.csv
```

---

## Results (verified, seed-0 reproducible)

Same 22 complete subjects × 9 questions = 198 rows.

| Feature set | mean per-Q accuracy | score-level r | score MAE |
|---|---|---|---|
| **Professor's f1/f2/f3** | **0.788** | **0.741** | **1.091** |
| Our Layer-1 SRI | 0.556 | 0.153 | 1.818 |
| baseline (per-Q majority) | 0.611 | — | — |

- **The professor's f1/f2/f3 reproduce the paper** (Table 2 range 0.725–0.875;
  thesis test r ≈ 0.74, MAE ≈ 1.0). This confirms Layer 2 (eqs. 7–9) is
  implemented correctly and that three state-proportions are enough to predict
  correctness.
- **Our own SRI is ≈ chance** → the open problem is **Layer-1 partition
  fidelity**: our K-medoids partition does not match the professor's. Every
  lever we can test on our side has been ruled out (see *Open blocker*).

---

## Repository layout

```
src/            pipeline source (run with: python src/<file>.py)
  config.py          constants: paths, fs, filter, channels, bands, STFT, K
  inspect_data.py    step 0 — confirm the EEG is band-pass filtered
  features.py        steps 1–3 — load, dedupe, preprocess, RSP features
  macrostates.py     steps 4–6 — K-medoids → states → SRI   (cluster_frames())
  spectral_power.py  deliverable #1 — per-state band power
  labels.py          load the professor's answer labels (f1/f2/f3 + correct/incorrect)
  classify.py        Layer 2 — per-question logistic regression + CV (eqs. 7–9)
  topomap.py         per-state band-power scalp maps
  plot_csv.py        SRI / band-power figures from the CSVs
  plot_layer2.py     per-question accuracy figure
  plot_student_groups.py  high- vs average-achiever SRI comparison

scripts/        throwaway read-only experiments (lever tests, sanity checks)
docs/           source paper + thesis (.docx), requirement, Thai summary,
                draft_for_professor.md (open questions), MEMORY.md (project log)
outputs/        tables/ (.csv), figures/ (.png), reports/ (.txt), experiments/
dataset/        iStar_data/  (EEG .mat per subject, + label_answer_data/*.xlsx)
```

---

## How to run

```bash
pip install -r requirements.txt          # numpy scipy pandas matplotlib scikit-learn mne

python src/macrostates.py     # → outputs/tables/sri.csv          (~1 min, K-medoids)
python src/spectral_power.py  # → outputs/tables/spectral_power.csv (deliverable #1)
python src/classify.py        # → outputs/tables/layer2_classification.csv (deliverables #2, #3)

python src/topomap.py             # → outputs/figures/topomaps.png
python src/plot_csv.py            # → SRI / band-power figures
python src/plot_layer2.py         # → per-question accuracy figure
python src/plot_student_groups.py # → achievement-group SRI figure
```

Each script is run from the repo root and prints a self-check (e.g. RSP/SRI rows
must sum to 1.0). The K-medoids seed is fixed (`default_rng(0)`), so
`sri.csv` regenerates byte-for-byte.

---

## Data notes (verified — see `docs/MEMORY.md` for the full log)

- EEG lives in `dataset/iStar_data/S1..S40/CutEEG_Cognitive/CutEEGdata_Q{q}_{seg}.mat`,
  a numeric `(samples, 19)` array (variable `CutEEGdata`). Multiple `_seg` files
  per question are cut segments → concatenated (after dedupe).
- The **9 visual–textual questions = Q1–Q9.** Q13/Q18 are the extra task → ignored.
- `fs = 1000 Hz`. Bands: θ 4–8, α 8–12, β 12–30 Hz.
- Cognitive `.mat` files are already 60 Hz-notched and mostly band-limited; the
  paper's filter is re-applied anyway (near-idempotent; cleans the under-filtered
  minority).
- **19-channel column order is assumed** standard 10-20 (from the montage photo
  `docs/IMG_0733.JPG`) — **unconfirmed** in the raw `.mat` (no channel labels
  exist in any file). 11 task channels used: Fp1, Fp2, F3, Fz, F4, C3, Cz, C4, P3, Pz, P4.
- 40 subjects total → 27 have complete Q1–Q9 files → **22** still complete after
  artifact rejection (S2, S16, S18, S25, S27 lose a question entirely).

---

## Open blocker & questions for the professor

The single open item is **Layer-1 partition fidelity** — our SRI is
uncorrelated with the professor's f1/f2/f3 (best |Pearson r| ≈ 0.06 ≈ chance),
and the 33-dim RSP cloud has no clean 3-cluster structure (silhouette ≈ 0.15,
no peak at K=3; K-medoids is seed-sensitive). Levers ruled out: channel order,
band-power-vs-macrostate, pooled-vs-per-question, 1-second epochs, standardization,
literal 0.1 s overlap, and question re-keying. So the next step is to obtain the
professor's **exact Layer-1 procedure** (K-medoids seed/init/restarts,
preprocessing, epoch definition — ideally their per-(subject, question) state
assignments).

Secondary asks: confirm the 19-channel column order; provide the 3-minute
resting baseline (needed for the paper's eq.10 change-ratio topomap); confirm
RSP (eq.2) is the intended Deliverable-#1 quantity. All framed in
`docs/draft_for_professor.md`.
