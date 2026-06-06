#!/usr/bin/env python3
"""Precompute DMD features on log grid (cache to disk for fast training)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_B = Path(__file__).resolve().parent
if str(_REPO_B) not in sys.path:
    sys.path.insert(0, str(_REPO_B))

from dmd_features import load_or_compute_dmd_grid  # noqa: E402
from taskB_dataset import load_ir_raw_from_npz  # noqa: E402
from taskB_grid import TaskBGrid, fit_target_norm, load_target_norm, save_target_norm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--norm_path", default=None, help="target_norm.json (fit if missing)")
    ap.add_argument("--n_bins", type=int, default=10000)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=10000.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    grid = TaskBGrid(fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins)
    if args.norm_path and Path(args.norm_path).exists():
        norm = load_target_norm(Path(args.norm_path))
    else:
        norm = fit_target_norm(data_dir, grid, max_files=300)
        if args.norm_path:
            save_target_norm(Path(args.norm_path), norm)

    irs = sorted(data_dir.glob("random_IR_[0-9]*.npz"))
    if args.limit > 0:
        irs = irs[: args.limit]

    for i, ir_path in enumerate(irs, start=1):
        fid = ir_path.stem.split("_")[-1]
        ir_raw, sr = load_ir_raw_from_npz(ir_path)
        load_or_compute_dmd_grid(
            ir_raw, sr, grid, norm, cache_dir, fid, recompute=args.recompute
        )
        if i == 1 or i % 10 == 0 or i == len(irs):
            print(f"[{i}/{len(irs)}] cached dmd_{fid}.npz", flush=True)

    print(f"Done. cache={cache_dir}")


if __name__ == "__main__":
    main()
