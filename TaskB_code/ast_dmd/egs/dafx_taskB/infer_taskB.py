#!/usr/bin/env python3
"""Infer Task B modal CSVs (supports scheme B: AST active + DMD sigma/gain)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_AST = Path(__file__).resolve().parents[2]
_EGS_B = Path(__file__).resolve().parent
for p in (str(_REPO_AST), str(_EGS_B)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fbank import FbankConfig  # noqa: E402
from ast_backbone import ASTLastClsBackbone  # noqa: E402
from taskB_dataset import DAFxTaskBDataset  # noqa: E402
from taskB_grid import TaskBGrid, load_target_norm, norm_to_gain, norm_to_sigma  # noqa: E402
from taskB_model import AstTaskBGridModel  # noqa: E402
from train_taskB import _model_inputs, collate_batch  # noqa: E402


def _resolve_param(
    idx: np.ndarray,
    b: int,
    *,
    source: str,
    dmd_valid,
    dmd_raw,
    net_row: np.ndarray,
    norm,
    to_raw,
) -> np.ndarray:
    out = np.empty(len(idx), dtype=np.float64)
    for j, bi in enumerate(idx):
        if source == "dmd" and dmd_valid is not None and dmd_valid[b, bi] > 0.5 and np.isfinite(dmd_raw[b, bi]):
            out[j] = float(dmd_raw[b, bi])
        else:
            out[j] = float(to_raw(np.array([net_row[bi]]), norm)[0])
    return out


@torch.no_grad()
def predict_modes_for_batch(
    model: AstTaskBGridModel,
    batch: dict,
    device: torch.device,
    grid: TaskBGrid,
    norm,
    active_thresh: float,
    scheme_b: bool,
    sigma_source: str = "dmd",
    gain_source: str = "dmd",
) -> list[pd.DataFrame]:
    model.eval()
    use_dmd = bool(batch.get("dmd_sigma_norm") is not None) or scheme_b
    kwargs = _model_inputs(batch, device, use_dmd)
    al, sn, gn = model(**kwargs)
    prob = torch.sigmoid(al).cpu().numpy()
    sn = sn.cpu().numpy()
    gn = gn.cpu().numpy()

    dmd_sigma = batch.get("dmd_sigma_norm")
    dmd_gain = batch.get("dmd_gain_norm")
    dmd_valid = batch.get("dmd_valid")
    dmd_sigma_raw = batch.get("dmd_sigma_raw")
    dmd_gain_raw = batch.get("dmd_gain_raw")
    if dmd_sigma is not None:
        dmd_sigma = dmd_sigma.cpu().numpy()
        dmd_gain = dmd_gain.cpu().numpy()
        dmd_valid = dmd_valid.cpu().numpy()
        dmd_sigma_raw = dmd_sigma_raw.cpu().numpy()
        dmd_gain_raw = dmd_gain_raw.cpu().numpy()

    centers = grid.centers_hz
    out_dfs = []
    for b in range(prob.shape[0]):
        act = prob[b] >= active_thresh
        idx = np.flatnonzero(act)
        if len(idx) == 0:
            out_dfs.append(pd.DataFrame(columns=["f0_ident", "sigma_ident", "gain_ident"]))
            continue
        f0 = centers[idx]

        use_dmd_params = scheme_b and dmd_valid is not None
        if use_dmd_params:
            sigma = _resolve_param(
                idx,
                b,
                source=sigma_source,
                dmd_valid=dmd_valid,
                dmd_raw=dmd_sigma_raw,
                net_row=sn[b],
                norm=norm,
                to_raw=norm_to_sigma,
            )
            gain = _resolve_param(
                idx,
                b,
                source=gain_source,
                dmd_valid=dmd_valid,
                dmd_raw=dmd_gain_raw,
                net_row=gn[b],
                norm=norm,
                to_raw=norm_to_gain,
            )
        else:
            sigma = norm_to_sigma(sn[b, idx], norm)
            gain = norm_to_gain(gn[b, idx], norm)

        out_dfs.append(
            pd.DataFrame(
                {"f0_ident": f0, "sigma_ident": sigma, "gain_ident": gain},
            )
        )
    return out_dfs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--active_thresh", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--dmd_cache_dir", default=None, help="Override DMD cache directory")
    ap.add_argument(
        "--sigma_source",
        choices=("dmd", "network"),
        default="dmd",
        help="Scheme B: sigma from DMD cache or network head (default: dmd)",
    )
    ap.add_argument(
        "--gain_source",
        choices=("dmd", "network"),
        default="dmd",
        help="Scheme B: gain from DMD cache or network head (default: dmd)",
    )
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    meta = json.loads((exp_dir / "meta.json").read_text(encoding="utf-8"))
    g = meta["grid"]
    grid = TaskBGrid(fmin=float(g["fmin"]), fmax=float(g["fmax"]), n_bins=int(g["n_bins"]))
    norm = load_target_norm(exp_dir / "target_norm.json")
    scheme_b = bool(meta.get("scheme_b", False))
    use_dmd = bool(meta.get("use_dmd_features", False))
    dmd_cache = args.dmd_cache_dir or meta.get("dmd_cache_dir")
    val_dmd_cache = meta.get("val_dmd_cache_dir", dmd_cache)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(exp_dir / args.ckpt), map_location="cpu")
    backbone = ASTLastClsBackbone(
        input_tdim=int(meta.get("target_length", 256)),
        audioset_ckpt_path=meta["audioset_ckpt"],
        freeze_backbone=True,
    )
    model = AstTaskBGridModel(
        backbone,
        n_bins=grid.n_bins,
        use_dmd_features=use_dmd,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    fbank_cfg = FbankConfig(
        target_length=int(meta.get("target_length", 256)),
        ir_amplitude_gain=float(meta.get("ir_amplitude_gain", 1e8)),
    )
    ds = DAFxTaskBDataset(
        args.data_dir,
        fbank_cfg,
        grid,
        norm,
        use_dmd=use_dmd,
        scheme_b=scheme_b,
        dmd_cache_dir=args.dmd_cache_dir or val_dmd_cache or dmd_cache,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for batch in loader:
        dfs = predict_modes_for_batch(
            model,
            batch,
            device,
            grid,
            norm,
            args.active_thresh,
            scheme_b,
            sigma_source=args.sigma_source,
            gain_source=args.gain_source,
        )
        for fid, df in zip(batch["file_id"], dfs):
            out_name = f"random_IR_identifiedModes_{fid}.csv"
            out_path = out_dir / out_name
            df.to_csv(out_path, index=False)
            summary.append({"file_id": fid, "rows": len(df), "out": str(out_path)})

    (out_dir / "_index_TaskB.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(summary)} files to {out_dir} "
        f"(scheme_b={scheme_b}, sigma_source={args.sigma_source}, gain_source={args.gain_source})"
    )


if __name__ == "__main__":
    main()
