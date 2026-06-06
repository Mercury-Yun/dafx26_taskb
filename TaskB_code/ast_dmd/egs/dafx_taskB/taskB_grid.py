"""Frequency grid utilities for Task B (log-spaced bins in [fmin, fmax])."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TaskBGrid:
    fmin: float = 20.0
    fmax: float = 10000.0
    n_bins: int = 4096

    @property
    def centers_hz(self) -> np.ndarray:
        return np.geomspace(self.fmin, self.fmax, self.n_bins).astype(np.float64)

    def bin_index_for_freq(self, f0_hz: np.ndarray) -> np.ndarray:
        c = self.centers_hz
        f0 = np.asarray(f0_hz, dtype=np.float64)
        # nearest log-bin
        logc = np.log(c)
        logf = np.log(np.clip(f0, c[0], c[-1]))
        idx = np.searchsorted(logc, logf, side="left")
        idx = np.clip(idx, 0, self.n_bins - 1)
        left = np.clip(idx - 1, 0, self.n_bins - 1)
        pick_left = np.abs(logf - logc[left]) <= np.abs(logf - logc[idx])
        return np.where(pick_left, left, idx).astype(np.int64)


@dataclass
class TaskBTargetNorm:
    """Normalization for sigma and log10(|gain|) on active bins only."""

    sigma_log_mean: float
    sigma_log_std: float
    log_gain_mean: float
    log_gain_std: float

    def to_json(self) -> dict:
        return self.__dict__.copy()

    @staticmethod
    def from_json(d: dict) -> "TaskBTargetNorm":
        return TaskBTargetNorm(**d)


def rasterize_modes_csv(
    modes_path: Path,
    grid: TaskBGrid,
    fmin: float | None = None,
    fmax: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (active, sigma_raw, gain_raw) each shape (n_bins,).
    Collisions: keep mode with largest |gain| in the bin.
    """
    fmin = grid.fmin if fmin is None else fmin
    fmax = grid.fmax if fmax is None else fmax

    # 尝试读取 CSV，使用 python 引擎提高容错性
    try:
        df = pd.read_csv(modes_path, engine='python')
    except Exception as e:
        print(f"Warning: Failed to read {modes_path}: {e}. Returning empty rasterization.")
        active = np.zeros(grid.n_bins, dtype=np.float32)
        sigma_out = np.zeros(grid.n_bins, dtype=np.float32)
        gain_out = np.zeros(grid.n_bins, dtype=np.float32)
        return active, sigma_out, gain_out

    # 检查必要的列是否存在
    required_cols = ['f0', 'sigma', 'gain']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: {modes_path} missing columns {required_cols}. Returning empty.")
        active = np.zeros(grid.n_bins, dtype=np.float32)
        sigma_out = np.zeros(grid.n_bins, dtype=np.float32)
        gain_out = np.zeros(grid.n_bins, dtype=np.float32)
        return active, sigma_out, gain_out

    f0 = df["f0"].values.astype(np.float64)
    sigma = df["sigma"].values.astype(np.float64)
    gain = df["gain"].values.astype(np.float64)
    mask = (f0 >= fmin) & (f0 <= fmax)
    f0, sigma, gain = f0[mask], sigma[mask], gain[mask]

    active = np.zeros(grid.n_bins, dtype=np.float32)
    sigma_out = np.zeros(grid.n_bins, dtype=np.float32)
    gain_out = np.zeros(grid.n_bins, dtype=np.float32)

    if len(f0) == 0:
        return active, sigma_out, gain_out

    bins = grid.bin_index_for_freq(f0)
    for i in range(len(f0)):
        b = int(bins[i])
        if active[b] < 1.0 or abs(gain[i]) > abs(gain_out[b]):
            active[b] = 1.0
            sigma_out[b] = float(sigma[i])
            gain_out[b] = float(gain[i])
    return active, sigma_out, gain_out


def ir_to_log_band_mag(
    ir: np.ndarray,
    sr: int,
    grid: TaskBGrid,
    eps: float = 1e-20,
) -> np.ndarray:
    """Log10 magnitude spectrum resampled to grid.centers_hz, shape (n_bins,)."""
    ir = np.asarray(ir, dtype=np.float64)
    spec = np.abs(np.fft.rfft(ir))
    freq = np.fft.rfftfreq(len(ir), d=1.0 / float(sr))
    mag = np.log10(np.maximum(spec, eps))
    # interpolate in log-frequency
    logf = np.log(np.clip(freq, grid.fmin, None))
    logc = np.log(grid.centers_hz)
    out = np.interp(logc, logf, mag, left=mag[0], right=mag[-1])
    return out.astype(np.float32)


def fit_target_norm(
    data_dir: Path,
    grid: TaskBGrid,
    max_files: int = 200,
) -> TaskBTargetNorm:
    """Estimate normalization from training modes CSVs."""
    mode_files = sorted(data_dir.glob("random_IR_modes_*.csv"))[:max_files]
    sig_logs: list[float] = []
    gain_logs: list[float] = []
    for p in mode_files:
        active, sigma, gain = rasterize_modes_csv(p, grid)
        act = active > 0.5
        s = sigma[act]
        g = gain[act]
        if len(s):
            sig_logs.extend(np.log(np.clip(s, 1e-12, None)).tolist())
        if len(g):
            gain_logs.extend(np.log10(np.clip(np.abs(g), 1e-30, None)).tolist())
    sig_logs_arr = np.array(sig_logs, dtype=np.float64)
    gain_logs_arr = np.array(gain_logs, dtype=np.float64)
    return TaskBTargetNorm(
        sigma_log_mean=float(sig_logs_arr.mean()) if len(sig_logs_arr) else 0.0,
        sigma_log_std=float(sig_logs_arr.std() + 1e-6),
        log_gain_mean=float(gain_logs_arr.mean()) if len(gain_logs_arr) else 0.0,
        log_gain_std=float(gain_logs_arr.std() + 1e-6),
    )


def save_target_norm(path: Path, norm: TaskBTargetNorm) -> None:
    path.write_text(json.dumps(norm.to_json(), indent=2), encoding="utf-8")


def load_target_norm(path: Path) -> TaskBTargetNorm:
    return TaskBTargetNorm.from_json(json.loads(path.read_text(encoding="utf-8")))


def sigma_to_norm(sigma: np.ndarray, norm: TaskBTargetNorm) -> np.ndarray:
    x = np.log(np.clip(sigma, 1e-12, None))
    return ((x - norm.sigma_log_mean) / norm.sigma_log_std).astype(np.float32)


def gain_to_norm(gain: np.ndarray, norm: TaskBTargetNorm) -> np.ndarray:
    x = np.log10(np.clip(np.abs(gain), 1e-30, None))
    return ((x - norm.log_gain_mean) / norm.log_gain_std).astype(np.float32)


def norm_to_sigma(sigma_n: np.ndarray, norm: TaskBTargetNorm) -> np.ndarray:
    x = sigma_n * norm.sigma_log_std + norm.sigma_log_mean
    return np.exp(x).astype(np.float64)


def norm_to_gain(gain_n: np.ndarray, norm: TaskBTargetNorm, sign: float = 1.0) -> np.ndarray:
    x = gain_n * norm.log_gain_std + norm.log_gain_mean
    return (sign * np.power(10.0, x)).astype(np.float64)