from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_AST = Path(__file__).resolve().parents[2]
_EGS_B = Path(__file__).resolve().parent
if str(_REPO_AST) not in sys.path:
    sys.path.insert(0, str(_REPO_AST))
if str(_EGS_B) not in sys.path:
    sys.path.insert(0, str(_EGS_B))

from fbank import FbankConfig, waveform_to_fbank  # noqa: E402

from taskB_grid import (  # noqa: E402
    TaskBGrid,
    TaskBTargetNorm,
    gain_to_norm,
    rasterize_modes_csv,
    sigma_to_norm,
)


def _modes_path_for_ir(ir_path: Path) -> Path:
    idx = ir_path.stem.split("_")[-1]
    return ir_path.with_name(f"random_IR_modes_{idx}.csv")


def load_ir_from_npz(npz_path: Path) -> Tuple[np.ndarray, int]:
    with np.load(str(npz_path)) as d:
        ir = np.asarray(d["ir"], dtype=np.float32)
        sr = int(d["sample_rate"])
        norm_fac = float(np.asarray(d["normalization_factor"]).reshape(()))
        peak = float(np.max(np.abs(ir))) if ir.size else 0.0
        scale = norm_fac if norm_fac > 1e-30 else peak
        if scale > 1e-30:
            ir = ir / scale
    return ir, sr


def load_ir_raw_from_npz(npz_path: Path) -> Tuple[np.ndarray, int]:
    """Unnormalized displacement IR (for DMD gain scale)."""
    with np.load(str(npz_path)) as d:
        ir = np.asarray(d["ir"], dtype=np.float64)
        sr = int(d["sample_rate"])
    return ir, sr


class DAFxTaskBDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fbank_cfg: FbankConfig,
        grid: TaskBGrid,
        target_norm: TaskBTargetNorm,
        use_dmd: bool = False,
        scheme_b: bool = False,
        dmd_cache_dir: str | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.fbank_cfg = fbank_cfg
        self.grid = grid
        self.norm = target_norm
        self.use_dmd = bool(use_dmd or scheme_b)
        self.scheme_b = bool(scheme_b)
        self.dmd_cache_dir = Path(dmd_cache_dir) if dmd_cache_dir else None

        irs = sorted(self.root_dir.glob("random_IR_[0-9]*.npz"))
        pairs: List[Tuple[Path, Path]] = []
        for ir in irs:
            modes = _modes_path_for_ir(ir)
            if modes.exists():
                pairs.append((ir, modes))
        if not pairs:
            raise RuntimeError(f"No npz/modes pairs under {self.root_dir}")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ir_path, modes_path = self.pairs[idx]
        file_id = ir_path.stem.split("_")[-1]
        ir, sr = load_ir_from_npz(ir_path)

        waveform = torch.from_numpy(ir).reshape(1, -1)
        x, log_e = waveform_to_fbank(waveform, sr, self.fbank_cfg)

        from taskB_grid import ir_to_log_band_mag

        band_mag = ir_to_log_band_mag(ir, sr, self.grid)
        active, sigma_raw, gain_raw = rasterize_modes_csv(modes_path, self.grid)

        sigma_n = sigma_to_norm(sigma_raw, self.norm)
        gain_n = gain_to_norm(gain_raw, self.norm)
        reg_mask = active.astype(np.float32)

        item = {
            "fbank": torch.as_tensor(x, dtype=torch.float32),
            "log_energy": log_e.to(dtype=torch.float32),
            "band_mag": torch.from_numpy(band_mag),
            "active": torch.from_numpy(active),
            "sigma_norm": torch.from_numpy(sigma_n),
            "gain_norm": torch.from_numpy(gain_n),
            "reg_mask": torch.from_numpy(reg_mask),
            "file_id": file_id,
        }

        if self.use_dmd:
            from dmd_features import load_or_compute_dmd_grid

            ir_raw, sr_raw = load_ir_raw_from_npz(ir_path)
            dmd = load_or_compute_dmd_grid(
                ir_raw,
                sr_raw,
                self.grid,
                self.norm,
                self.dmd_cache_dir,
                file_id,
            )
            item["dmd_sigma_norm"] = torch.from_numpy(dmd["dmd_sigma_norm"])
            item["dmd_gain_norm"] = torch.from_numpy(dmd["dmd_gain_norm"])
            item["dmd_valid"] = torch.from_numpy(dmd["dmd_valid"])
            item["dmd_sigma_raw"] = torch.from_numpy(dmd["dmd_sigma_raw"])
            item["dmd_gain_raw"] = torch.from_numpy(dmd["dmd_gain_raw"])

        return item
