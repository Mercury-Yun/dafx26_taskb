#!/usr/bin/env python3
"""Inference on a folder containing only IR .npz files (no ground truth CSV)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_REPO_AST = Path(__file__).resolve().parents[2]
_EGS_B = Path(__file__).resolve().parent
for p in (str(_REPO_AST), str(_EGS_B)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fbank import FbankConfig, waveform_to_fbank
from ast_backbone import ASTLastClsBackbone
from taskB_dataset import load_ir_raw_from_npz, load_ir_from_npz
from taskB_grid import TaskBGrid, load_target_norm, norm_to_gain, norm_to_sigma, ir_to_log_band_mag
from taskB_model import AstTaskBGridModel
from dmd_features import load_or_compute_dmd_grid, DMDConfig


class InferenceOnlyDataset(Dataset):
    """Dataset for inference on folders containing only .npz files (no modes CSV)."""
    def __init__(
        self,
        root_dir: str,
        fbank_cfg: FbankConfig,
        grid: TaskBGrid,
        target_norm,
        use_dmd: bool = False,
        dmd_cache_dir: str | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.fbank_cfg = fbank_cfg
        self.grid = grid
        self.target_norm = target_norm
        self.use_dmd = use_dmd
        self.dmd_cache_dir = Path(dmd_cache_dir) if dmd_cache_dir else None

        self.ir_files = sorted(self.root_dir.glob("random_IR_*.npz"))
        if not self.ir_files:
            raise RuntimeError(f"No random_IR_*.npz files found in {self.root_dir}")

    def __len__(self):
        return len(self.ir_files)

    def __getitem__(self, idx):
        ir_path = self.ir_files[idx]
        file_id = ir_path.stem.split("_")[-1]

        # Load normalized IR for AST and band_mag
        ir, sr = load_ir_from_npz(ir_path)
        waveform = torch.from_numpy(ir).reshape(1, -1)
        x, log_e = waveform_to_fbank(waveform, sr, self.fbank_cfg)

        band_mag = ir_to_log_band_mag(ir, sr, self.grid)  # (n_bins,)

        item = {
            "fbank": torch.as_tensor(x, dtype=torch.float32),
            "log_energy": log_e.to(dtype=torch.float32),
            "band_mag": torch.from_numpy(band_mag),
            "file_id": file_id,
        }

        if self.use_dmd:
            # Raw IR for DMD (unnormalized)
            ir_raw, sr_raw = load_ir_raw_from_npz(ir_path)
            dmd = load_or_compute_dmd_grid(
                ir_raw,
                sr_raw,
                self.grid,
                self.target_norm,
                self.dmd_cache_dir,
                file_id,
                cfg=DMDConfig(),
                recompute=False,
            )
            item["dmd_sigma_norm"] = torch.from_numpy(dmd["dmd_sigma_norm"])
            item["dmd_gain_norm"] = torch.from_numpy(dmd["dmd_gain_norm"])
            item["dmd_valid"] = torch.from_numpy(dmd["dmd_valid"])
            item["dmd_sigma_raw"] = torch.from_numpy(dmd["dmd_sigma_raw"])
            item["dmd_gain_raw"] = torch.from_numpy(dmd["dmd_gain_raw"])

        return item


def collate_batch(samples):
    out = {}
    for k in samples[0].keys():
        if k == "file_id":
            out[k] = [s[k] for s in samples]
        else:
            out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out


def _model_inputs(batch, device, use_dmd):
    fbank = batch["fbank"].to(device)
    band_mag = batch["band_mag"].to(device)
    log_e = batch["log_energy"].to(device)
    if not use_dmd:
        return dict(fbank=fbank, band_mag=band_mag, log_energy=log_e)
    return dict(
        fbank=fbank,
        band_mag=band_mag,
        log_energy=log_e,
        dmd_sigma_norm=batch["dmd_sigma_norm"].to(device),
        dmd_gain_norm=batch["dmd_gain_norm"].to(device),
        dmd_valid=batch["dmd_valid"].to(device),
    )


@torch.no_grad()
def predict_modes_for_batch(model, batch, device, grid, norm, active_thresh):
    model.eval()
    use_dmd = "dmd_sigma_norm" in batch
    kwargs = _model_inputs(batch, device, use_dmd)
    al, sn, gn = model(**kwargs)
    prob = torch.sigmoid(al).cpu().numpy()
    sn = sn.cpu().numpy()
    gn = gn.cpu().numpy()

    centers = grid.centers_hz
    out_dfs = []
    for b in range(prob.shape[0]):
        act = prob[b] >= active_thresh
        idx = np.flatnonzero(act)
        if len(idx) == 0:
            out_dfs.append(pd.DataFrame(columns=["f0_ident", "sigma_ident", "gain_ident"]))
            continue
        f0 = centers[idx]
        sigma = norm_to_sigma(sn[b, idx], norm)
        gain = norm_to_gain(gn[b, idx], norm)
        out_dfs.append(pd.DataFrame({"f0_ident": f0, "sigma_ident": sigma, "gain_ident": gain}))
    return out_dfs


def main():
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--ckpt", default="best.pt")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--active_thresh", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dmd_cache_dir", default=None)
    args = parser.parse_args()
    t_start = time.time()

    exp_dir = Path(args.exp_dir)
    meta = json.loads((exp_dir / "meta.json").read_text())
    g = meta["grid"]
    grid = TaskBGrid(fmin=g["fmin"], fmax=g["fmax"], n_bins=g["n_bins"])
    norm = load_target_norm(exp_dir / "target_norm.json")
    use_dmd = meta.get("use_dmd_features", False)
    dmd_cache = args.dmd_cache_dir or meta.get("dmd_cache_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(exp_dir / args.ckpt, map_location="cpu")
    backbone = ASTLastClsBackbone(
        input_tdim=meta.get("target_length", 256),
        audioset_ckpt_path=meta["audioset_ckpt"],
        freeze_backbone=True,
    )
    model = AstTaskBGridModel(backbone, n_bins=grid.n_bins, use_dmd_features=use_dmd).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    fbank_cfg = FbankConfig(
        target_length=meta.get("target_length", 256),
        ir_amplitude_gain=meta.get("ir_amplitude_gain", 1e8),
    )
    ds = InferenceOnlyDataset(
        args.data_dir, fbank_cfg, grid, norm,
        use_dmd=use_dmd, dmd_cache_dir=dmd_cache
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_batch)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for batch in loader:
        dfs = predict_modes_for_batch(model, batch, device, grid, norm, args.active_thresh)
        for fid, df in zip(batch["file_id"], dfs):
            # Official Task B submission naming (see baseline TaskB/README.md)
            out_path = out_dir / f"random_IR_identifiedModes_{fid}.csv"
            df.to_csv(out_path, index=False)
            summary.append({"file_id": fid, "rows": len(df), "out": str(out_path)})
    elapsed = time.time() - t_start
    index = {"files": summary, "n_files": len(summary), "inference_seconds": elapsed}
    (out_dir / "_index_TaskB.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {len(summary)} files to {out_dir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()