# DAFx26 Task B Submission

Task B solution: dual-stream **AST + DMD** model on a 10,000-bin log-frequency grid (20 Hz–10 kHz).

## Package contents

| Item | Path |
|------|------|
| Method write-up | `Technique_Report_for_TaskB.pdf` |
| Reproduction guide | `TaskB_code/README.md` |
| Source code (AST + DMD) | `TaskB_code/ast_dmd/` (includes AudioSet backbone weights) |
| Test-set predictions | `TaskB_results/random_IR_identifiedModes_*.csv` |
| Runtime metadata | `TaskB_runtime/epochs.txt`, `compute_time.txt`, `hardware_description.txt` |

## Reproduction (summary)

See **`TaskB_code/README.md`** for full commands. Overview:

0. **Data** — train/val from official `ModalPlate.DatasetGen`; test set from `2026-DATASET-STRIPPED`
1. Install dependencies (`TaskB_code/requirements.txt`)
2. Precompute DMD cache for train / val / test
3. Train 100 epochs (Focal + Huber + frequency weighting; hyperparameters in the code README)
4. Select `active_thresh = 0.411` on the validation set (`infer_taskB.py` + `eval_fast.py`)
5. Run `infer_test.py` on the official test set → `TaskB_results/`

Train/validation IRs are synthesized with the challenge **ModalPlate** code; the test IRs are from the public **`2026-DATASET-STRIPPED`** release (details in `TaskB_code/README.md` § Dataset).
