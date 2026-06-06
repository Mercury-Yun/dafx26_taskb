# Task B — Reproduction Guide (AST + DMD)

Dual-stream model: **AST** (16 kHz log-mel) + **DMD / log-spectrum** features → per-bin **activity**, **σ**, and **gain** on a **10,000-bin** log grid (20 Hz–10 kHz).

This document describes how to reproduce the submitted Task B experiment end to end.

---

## Training configuration

| Parameter | Value |
|-----------|-------|
| Activity loss | Focal (`--loss_active focal`, γ=2, α=0.25) |
| Regression loss | Huber (`--loss_reg huber`, δ=1.0) |
| Low-frequency weight | `--freq_weight` (extra weight for bins &lt; 200 Hz) |
| `pos_weight` | 5.0 |
| `w_sigma`, `w_gain` | 3.0 |
| `epochs` | 100 |
| `batch_size` | 4 |
| `lr` / `backbone_lr` | 1e-4 / 5e-6 |
| Grid | 10,000 bins, 20–10,000 Hz |
| Backbone | AudioSet AST, unfrozen (`--unfreeze_backbone`) |
| DMD features | `--use_dmd_features` |
| Inference threshold | **`active_thresh = 0.411`** (selected on validation set) |

**Included:** `pretrained_models/audioset_10_10_0.4593.pth` (~337 MB).

**Required externally:** training/validation/test IR data (see **Dataset** below).

---

## Dataset

All impulse responses are **synthetic plate IRs** from the official DAFx challenge **ModalPlate** simulator (`ModalPlate/ModalPlate.py`, generator `ModalPlate/DatasetGen.py` — e.g. `baseline_latest/ModalPlate` in the challenge baseline repo).

For each plate draw, `DatasetGen` randomizes physical parameters (ranges in `ParamRange.py`), synthesizes the displacement IR, and writes:

| File | Use in Task B |
|------|----------------|
| `random_IR_XXXX.npz` | **Input** — unnormalized displacement + metadata (`ir`, `sample_rate`, …). **Use this for training and inference.** |
| `random_IR_modes_XXXX.csv` | **Supervision** — ground-truth `(f0, sigma, gain)` per mode (train/val only). |
| `random_IR_XXXX.wav` | Listening only (peak-normalized; **do not** use for Task B). |
| `random_IR_params_XXXX.csv` | Task A ground truth (not needed for Task B). |

**This submission uses three splits:**

| Split | Source | Size (our run) | IR length | Modes CSV |
|-------|--------|----------------|-----------|-----------|
| **Train** | Generated locally with `DatasetGen` | 10,000 | 1.0 s @ 44.1 kHz | yes |
| **Validation** | Generated locally with `DatasetGen` | 1,000 | 1.0 s @ 44.1 kHz | yes |
| **Test** | Official release folder `2026-DATASET-STRIPPED` (organizers) | 16 | 5.0 s @ 44.1 kHz | no (held out for scoring) |

Obtain the **test set** from the challenge dataset release. Regenerate **train/val** (or equivalent splits) with the official generator, e.g. from the challenge repo root:

```bash
pip install -r requirements.txt   # challenge repo requirements (includes soundfile, etc.)

# Training set
python3 -m ModalPlate.DatasetGen --number 10000 --duration 1.0 --out /path/to/train_ds

# Validation set (use a different --seed if you want a disjoint draw)
python3 -m ModalPlate.DatasetGen --number 1000 --duration 1.0 --out /path/to/val_ds --seed 12345
```

Point `TRAIN_DIR`, `VAL_DIR`, and `TEST_DIR` in Step 0 to these folders.  
**Important:** use `.npz` displacement data only; do not train on peak-normalized `.wav` files.

---

## Folder layout (this submission)

```
TaskB/
├── Technique_Report_for_TaskB.pdf
├── README.md                      ← submission overview
├── TaskB_code/
│   ├── requirements.txt
│   ├── README.md                  ← this file (full reproduction steps)
│   └── ast_dmd/                   ← AST + DMD dual-stream implementation
│       ├── pretrained_models/audioset_10_10_0.4593.pth
│       ├── src/models/ast_models.py
│       └── egs/dafx_taskB/        ← train / infer / eval scripts
├── TaskB_results/                 ← test-set submission CSVs
└── TaskB_runtime/                 ← epochs.txt, compute_time.txt, hardware_description.txt
```

---

## Step 0 — Set paths

Edit and export these before running any command:

```bash
# Unpack TaskB/, then:
export TASKB_ROOT="/path/to/TaskB"
export CODE_ROOT="${TASKB_ROOT}/TaskB_code/ast_dmd"
cd "${CODE_ROOT}"

# Official dataset (see “Dataset” section)
export TRAIN_DIR="/path/to/train_ds"    # 10k IRs: .npz + random_IR_modes_*.csv
export VAL_DIR="/path/to/val_ds"        # 1k IRs: .npz + modes CSV
export TEST_DIR="/path/to/2026-DATASET-STRIPPED"  # 16 official test .npz (no modes CSV)

# Experiment output (created by you)
export EXP_DIR="/path/to/exp"
export DMD_CACHE="${EXP_DIR}/dmd_cache"
mkdir -p "${EXP_DIR}" "${DMD_CACHE}/train" "${DMD_CACHE}/val" "${DMD_CACHE}/test"

# Bundled backbone (relative to CODE_ROOT)
export AUDIOSET_CKPT="pretrained_models/audioset_10_10_0.4593.pth"
```

**Data rules**

- Use **unnormalized displacement IR inside `.npz`** for DMD (field `ir`); do not use peak-normalized `.wav` for training features.
- Train/val: each `random_IR_XXXX.npz` needs matching `random_IR_modes_XXXX.csv`.

---

## Step 1 — Install dependencies

```bash
cd "${CODE_ROOT}"
pip install -r ../requirements.txt
```

Tested: Python 3.8+, PyTorch 1.8.1, torchaudio 0.8.1, timm **0.4.5** (required by `ast_models.py`), scipy, pandas. Single GPU with ≥24 GB VRAM recommended (see `../TaskB_runtime/hardware_description.txt`).

---

## Step 2 — Precompute DMD cache

Required before training with `--use_dmd_features`. This step also creates `target_norm.json` if it does not exist.

```bash
cd "${CODE_ROOT}"

python3 egs/dafx_taskB/precompute_dmd_cache.py \
  --data_dir "${TRAIN_DIR}" \
  --cache_dir "${DMD_CACHE}/train" \
  --norm_path "${EXP_DIR}/target_norm.json" \
  --n_bins 10000 --fmin 20 --fmax 10000

python3 egs/dafx_taskB/precompute_dmd_cache.py \
  --data_dir "${VAL_DIR}" \
  --cache_dir "${DMD_CACHE}/val" \
  --norm_path "${EXP_DIR}/target_norm.json" \
  --n_bins 10000 --fmin 20 --fmax 10000
```

Optional (speeds up test inference; otherwise DMD is computed on the fly):

```bash
python3 egs/dafx_taskB/precompute_dmd_cache.py \
  --data_dir "${TEST_DIR}" \
  --cache_dir "${DMD_CACHE}/test" \
  --norm_path "${EXP_DIR}/target_norm.json" \
  --n_bins 10000 --fmin 20 --fmax 10000
```

---

## Step 3 — Train

```bash
cd "${CODE_ROOT}"

python3 egs/dafx_taskB/train_taskB.py \
  --train_dir "${TRAIN_DIR}" \
  --val_dir "${VAL_DIR}" \
  --audioset_ckpt "${AUDIOSET_CKPT}" \
  --exp_dir "${EXP_DIR}" \
  --use_dmd_features \
  --dmd_cache_dir "${DMD_CACHE}/train" \
  --val_dmd_cache_dir "${DMD_CACHE}/val" \
  --unfreeze_backbone \
  --epochs 100 \
  --batch_size 4 \
  --num_workers 2 \
  --n_bins 10000 \
  --fmin 20 \
  --fmax 10000 \
  --lr 1e-4 \
  --backbone_lr 5e-6 \
  --pos_weight 5.0 \
  --w_sigma 3.0 \
  --w_gain 3.0 \
  --loss_active focal \
  --loss_reg huber \
  --freq_weight
```

**Outputs in `${EXP_DIR}`:** `best.pt`, `last.pt`, `meta.json`, `history.json`, `target_norm.json`, `compute_summary.json`.

**Resume** (if interrupted):

```bash
python3 egs/dafx_taskB/train_taskB.py \
  ...same flags as above... \
  --resume "${EXP_DIR}/last.pt"
```

**Optional — lower learning rate for continued training:**

```bash
python3 egs/dafx_taskB/train_taskB.py \
  ...same flags... \
  --lr 1e-5 \
  --backbone_lr 1e-6 \
  --resume "${EXP_DIR}/last.pt"
```

> **Note:** Pass all flags above explicitly. Script defaults differ (e.g. `epochs=30`). Recorded training: 100 epochs, ~14.7 h (`../TaskB_runtime/`).

---

## Step 4 — Validation inference and threshold selection

Tune `--active_thresh` on the validation set; the submitted model uses **0.411**.

```bash
cd "${CODE_ROOT}"

python3 egs/dafx_taskB/infer_taskB.py \
  --data_dir "${VAL_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --ckpt best.pt \
  --out_dir "${EXP_DIR}/predictions_val" \
  --batch_size 4 \
  --active_thresh 0.411 \
  --sigma_source network \
  --gain_source network

python3 egs/dafx_taskB/eval_fast.py \
  --experiment_folder "${EXP_DIR}/predictions_val" \
  --target_folder "${VAL_DIR}" \
  --fmin 20 --fmax 10000 \
  --output "${EXP_DIR}/predictions_val/evaluation_fast.csv"
```

`eval_fast.py` uses greedy mode matching (suitable for 10k bins). Select the threshold with the lowest mean RE on validation before test inference.

---

## Step 5 — Test-set inference (submitted results)

```bash
cd "${CODE_ROOT}"

python3 egs/dafx_taskB/infer_test.py \
  --data_dir "${TEST_DIR}" \
  --exp_dir "${EXP_DIR}" \
  --ckpt best.pt \
  --out_dir "${EXP_DIR}/predictions_test" \
  --active_thresh 0.411 \
  --dmd_cache_dir "${DMD_CACHE}/test"
```

Each output file: `random_IR_identifiedModes_XXXX.csv` with columns `f0_ident`, `sigma_ident`, `gain_ident`.

---

## Script reference

| Script | Role |
|--------|------|
| `precompute_dmd_cache.py` | Precompute DMD features on disk |
| `train_taskB.py` | Training (Focal + Huber + frequency weighting) |
| `infer_taskB.py` | Validation-set inference |
| `infer_test.py` | Test-set inference → official CSV names |
| `eval_fast.py` | Fast RE evaluation on validation set |
| `dmd_features.py`, `taskB_*.py`, `fbank.py`, `ast_backbone.py` | Model and data pipeline |
| `src/models/ast_models.py` | AST backbone (Gong et al.) |

---

Training runtime metadata (see `../TaskB_runtime/`):

- `epochs.txt` → 100  
- `compute_time.txt` → 14.7 h total training  
- `hardware_description.txt` → GPU, CPU, RAM, OS
