"""DMD features on STFT, resampled to Task B log-frequency grid."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import fft

from taskB_grid import TaskBGrid, gain_to_norm, sigma_to_norm, TaskBTargetNorm


@dataclass(frozen=True)
class DMDConfig:
    frmsize: int = 2048
    noverlap: int = 4
    band_neighborhood: int = 15
    rank: int = 3
    time_trim: int = 3
    stft_stride: int = 2
    target_sr: int = 48000


def _kaiser_window(winsize: int, noverlap: int) -> np.ndarray:
    from scipy.signal import windows

    kwin = windows.kaiser(winsize, beta=1.9 * np.pi)
    kwsigma = np.sqrt(np.sum(kwin**2) / winsize * noverlap)
    return (kwin / kwsigma).astype(np.float64)


def implement_stft(x_input_signal: np.ndarray, frmsize: int, noverlap: int = 4) -> tuple[np.ndarray, dict]:
    x = np.asarray(x_input_signal, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    K, M = x.shape
    winsize = frmsize * noverlap
    nfbands = winsize // 2 + 1
    kwin = _kaiser_window(winsize, noverlap)

    nfrms = int(np.ceil(K / frmsize))
    Kfrm = nfrms * frmsize
    if Kfrm > K:
        x = np.vstack([x, np.zeros((Kfrm - K, M))])

    inBuf = np.zeros((winsize, M), dtype=np.float64)
    X_output_matrix = np.zeros((nfbands, nfrms), dtype=np.complex128)

    for nf in range(nfrms):
        start = nf * frmsize
        end = (nf + 1) * frmsize
        yfrm1 = x[start:end, 0]
        inBuf[:, 0] = np.concatenate([inBuf[frmsize:, 0], yfrm1])
        inWin1 = fft(inBuf[:, 0] * kwin)
        X_output_matrix[:, nf] = inWin1[:nfbands]

    stft_parameters = {
        "frmsize": frmsize,
        "noverlap": noverlap,
        "winsize": winsize,
        "nfbands": nfbands,
        "sample_rate": None,
    }
    return X_output_matrix, stft_parameters


def _dmd_at_band(Y_matrix: np.ndarray, rank: int, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (sigma, gain, valid) length-R arrays for one frequency band."""
    if Y_matrix.shape[1] < 3:
        return np.full(rank, np.nan), np.full(rank, np.nan), np.zeros(rank, dtype=bool)

    Y1 = Y_matrix[:, :-1]
    Y2 = Y_matrix[:, 1:]
    Uf, s, Vh = np.linalg.svd(Y1, full_matrices=False)
    if len(s) < rank or s[rank - 1] < 1e-12:
        return np.full(rank, np.nan), np.full(rank, np.nan), np.zeros(rank, dtype=bool)

    Vf = Vh.conj().T
    Ur = Uf[:, :rank]
    Sr_inv = np.diag(1.0 / s[:rank])
    Vr = Vf[:, :rank]
    Atilde = Ur.conj().T @ Y2 @ Vr @ Sr_inv
    eigvals, W = np.linalg.eig(Atilde)
    Phi = Y2 @ Vr @ Sr_inv @ W
    x1 = Y1[:, 0]
    b, _, _, _ = np.linalg.lstsq(Phi, x1, rcond=None)

    omega = np.log(eigvals) / dt
    sigma = np.maximum(-np.real(omega), 0.0)
    gain = np.abs(b)
    valid = np.isfinite(sigma) & np.isfinite(gain) & (sigma > 0)
    return sigma, gain, valid


def compute_dmd_on_stft_bins(
    ir: np.ndarray,
    sr: int,
    cfg: DMDConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run DMD on subsampled STFT frequency bins.

    Returns
    -------
    stft_freq_hz : (F,)
    sigma_fb : (F, R)
    gain_fb : (F, R)
    valid_fb : (F, R)
    """
    from scipy.signal import resample_poly
    from fractions import Fraction

    ir = np.asarray(ir, dtype=np.float64).reshape(-1)
    if sr != cfg.target_sr:
        frac = Fraction(cfg.target_sr, sr).limit_denominator(10000)
        ir = resample_poly(ir, frac.numerator, frac.denominator).astype(np.float64)
        sr = cfg.target_sr

    Y_stft, stft_params = implement_stft(ir, cfg.frmsize, cfg.noverlap)
    winsize = stft_params["winsize"]
    nfbands = stft_params["nfbands"]
    nfrms = Y_stft.shape[1]
    stft_freq_hz = np.arange(nfbands) * sr / winsize
    dt = cfg.frmsize / float(sr)

    N = cfg.band_neighborhood
    t0, t1 = cfg.time_trim, nfrms - cfg.time_trim
    if t1 - t0 < 4:
        empty = np.zeros((0, cfg.rank), dtype=np.float64)
        return stft_freq_hz[:0], empty, empty, empty.astype(bool)

    f_indices = np.arange(N, nfbands - N, cfg.stft_stride, dtype=np.int64)
    sigma_fb = np.full((len(f_indices), cfg.rank), np.nan, dtype=np.float64)
    gain_fb = np.full((len(f_indices), cfg.rank), np.nan, dtype=np.float64)
    valid_fb = np.zeros((len(f_indices), cfg.rank), dtype=bool)

    for j, f_index in enumerate(f_indices):
        Y_matrix = Y_stft[f_index - N : f_index + N + 1, t0:t1]
        s, g, v = _dmd_at_band(Y_matrix, cfg.rank, dt)
        sigma_fb[j] = s
        gain_fb[j] = g
        valid_fb[j] = v

    return stft_freq_hz[f_indices], sigma_fb, gain_fb, valid_fb


def _best_mode(sigma_r: np.ndarray, gain_r: np.ndarray, valid_r: np.ndarray) -> tuple[float, float, bool]:
    if not valid_r.any():
        return np.nan, np.nan, False
    idx = np.argmax(gain_r * valid_r)
    if not valid_r[idx]:
        idx = int(np.flatnonzero(valid_r)[0])
    return float(sigma_r[idx]), float(gain_r[idx]), True


def dmd_to_grid(
    ir: np.ndarray,
    sr: int,
    grid: TaskBGrid,
    norm: TaskBTargetNorm,
    cfg: DMDConfig | None = None,
) -> dict[str, np.ndarray]:
    """Map DMD STFT results onto log-spaced grid bins."""
    cfg = cfg or DMDConfig()
    stft_freq, sigma_fb, gain_fb, valid_fb = compute_dmd_on_stft_bins(ir, sr, cfg)

    n_bins = grid.n_bins
    sigma_raw = np.full(n_bins, np.nan, dtype=np.float32)
    gain_raw = np.full(n_bins, np.nan, dtype=np.float32)
    valid = np.zeros(n_bins, dtype=np.float32)

    if len(stft_freq) == 0:
        return _pack_dmd_grid(sigma_raw, gain_raw, valid, norm)

    centers = grid.centers_hz
    logc = np.log(centers)
    logf = np.log(np.clip(stft_freq, grid.fmin, None))

    for j in range(len(stft_freq)):
        s, g, ok = _best_mode(sigma_fb[j], gain_fb[j], valid_fb[j])
        if not ok:
            continue
        # assign to nearest log bin
        lf = logf[j]
        idx = int(np.argmin(np.abs(logc - lf)))
        if not np.isfinite(sigma_raw[idx]) or (np.isfinite(g) and g > abs(gain_raw[idx])):
            sigma_raw[idx] = np.float32(s)
            gain_raw[idx] = np.float32(g)
            valid[idx] = 1.0

    return _pack_dmd_grid(sigma_raw, gain_raw, valid, norm)


def _pack_dmd_grid(
    sigma_raw: np.ndarray,
    gain_raw: np.ndarray,
    valid: np.ndarray,
    norm: TaskBTargetNorm,
) -> dict[str, np.ndarray]:
    sigma_norm = np.zeros_like(sigma_raw, dtype=np.float32)
    gain_norm = np.zeros_like(gain_raw, dtype=np.float32)
    m = valid > 0.5
    if m.any():
        sigma_norm[m] = sigma_to_norm(sigma_raw[m], norm)
        gain_norm[m] = gain_to_norm(gain_raw[m], norm)
    return {
        "dmd_sigma_raw": sigma_raw,
        "dmd_gain_raw": gain_raw,
        "dmd_sigma_norm": sigma_norm,
        "dmd_gain_norm": gain_norm,
        "dmd_valid": valid.astype(np.float32),
    }


def dmd_cache_path(cache_dir: Path, file_id: str) -> Path:
    return cache_dir / f"dmd_{file_id}.npz"


def load_or_compute_dmd_grid(
    ir: np.ndarray,
    sr: int,
    grid: TaskBGrid,
    norm: TaskBTargetNorm,
    cache_dir: Path | None,
    file_id: str,
    cfg: DMDConfig | None = None,
    recompute: bool = False,
) -> dict[str, np.ndarray]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = dmd_cache_path(cache_dir, file_id)
        if path.exists() and not recompute:
            with np.load(path) as d:
                return {k: np.asarray(d[k]) for k in d.files}

    out = dmd_to_grid(ir, sr, grid, norm, cfg=cfg)

    if cache_dir is not None:
        np.savez_compressed(path, **out)
    return out
