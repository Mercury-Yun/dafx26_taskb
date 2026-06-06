#!/usr/bin/env python3
"""Fast evaluation for Task B using greedy matching (O(N log N))."""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# ---------- column mappings (same as original eval.py) ----------
GT_FREQ_COLS = ["f0", "freq", "frequency", "f0_actual"]
GT_SIGMA_COLS = ["sigma", "sigma_actual", "decay", "damping"]
GT_GAIN_COLS = ["gain", "gain_actual", "amplitude", "b", "modal_gain"]
EST_FREQ_COLS = ["f0_ident", "f0_identified", "f0_est", "f0_estimated", "freq_ident"]
EST_SIGMA_COLS = ["sigma_ident", "sigma_identified", "sigma_est", "sigma_estimated"]
EST_GAIN_COLS = ["gain_ident", "gain_identified", "gain_est", "gain_estimated"]

def _get_col(df, candidates):
    df_cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in df_cols_lower:
            return df_cols_lower[cand.lower()]
    return None

def load_ground_truth(csv_path, fmin=0.0, fmax=np.inf):
    df = pd.read_csv(csv_path)
    freq_col = _get_col(df, GT_FREQ_COLS)
    sigma_col = _get_col(df, GT_SIGMA_COLS)
    gain_col = _get_col(df, GT_GAIN_COLS)
    if freq_col is None or sigma_col is None or gain_col is None:
        raise ValueError(f"Missing columns in {csv_path}")
    f0 = df[freq_col].values.astype(float)
    sigma = df[sigma_col].values.astype(float)
    gain = df[gain_col].values.astype(float)
    mask = (f0 >= fmin) & (f0 <= fmax)
    return f0[mask], sigma[mask], gain[mask]

def load_estimated(csv_path):
    df = pd.read_csv(csv_path)
    freq_col = _get_col(df, EST_FREQ_COLS)
    sigma_col = _get_col(df, EST_SIGMA_COLS)
    gain_col = _get_col(df, EST_GAIN_COLS)
    if freq_col is None or sigma_col is None or gain_col is None:
        raise ValueError(f"Missing columns in {csv_path}")
    f0 = df[freq_col].values.astype(float)
    sigma = df[sigma_col].values.astype(float)
    gain = df[gain_col].values.astype(float)
    valid = np.isfinite(f0) & np.isfinite(sigma) & np.isfinite(gain)
    return f0[valid], sigma[valid], gain[valid]

def match_modes_greedy(f0_actual, f0_ident, match_threshold_octaves=0.5):
    """Greedy matching by frequency (fast)."""
    M = len(f0_actual)
    M_tilde = len(f0_ident)
    if M == 0:
        return [], [], list(range(M_tilde))
    if M_tilde == 0:
        return [], list(range(M)), []
    eps = 1e-12
    # sort by log2 frequency
    la = np.log2(np.maximum(f0_actual, eps))
    li = np.log2(np.maximum(f0_ident, eps))
    order_a = np.argsort(la)
    order_i = np.argsort(li)
    la_sorted = la[order_a]
    li_sorted = li[order_i]
    i, j = 0, 0
    matches = []
    used_a = set()
    used_i = set()
    while i < M and j < M_tilde:
        dist = abs(la_sorted[i] - li_sorted[j])
        if dist <= match_threshold_octaves:
            matches.append((order_a[i], order_i[j]))
            used_a.add(order_a[i])
            used_i.add(order_i[j])
            i += 1
            j += 1
        elif la_sorted[i] < li_sorted[j]:
            i += 1
        else:
            j += 1
    unmatched_a = sorted(set(range(M)) - used_a)
    unmatched_i = sorted(set(range(M_tilde)) - used_i)
    return matches, unmatched_a, unmatched_i

def compute_metrics(f0_a, sigma_a, gain_a, f0_i, sigma_i, gain_i):
    M = len(f0_a)
    M_tilde = len(f0_i)
    delta_M = abs(M - M_tilde)
    if M == 0:
        return {"RE_omega":0.0,"RE_sigma":0.0,"RE_gain":0.0,"RE0":0.0,"delta_M":delta_M,"RE":float(delta_M),
                "M_actual":M,"M_identified":M_tilde,"n_matched":0}
    matches, unmatched_a, _ = match_modes_greedy(f0_a, f0_i)
    re_omega = np.ones(M)
    re_sigma = np.ones(M)
    re_gain = np.ones(M)
    for a_idx, i_idx in matches:
        re_omega[a_idx] = min(1.0, abs(f0_i[i_idx]-f0_a[a_idx])/max(abs(f0_a[a_idx]),1e-30))
        re_sigma[a_idx] = min(1.0, abs(sigma_i[i_idx]-sigma_a[a_idx])/max(abs(sigma_a[a_idx]),1e-30))
        re_gain[a_idx] = min(1.0, abs(gain_i[i_idx]-gain_a[a_idx])/max(abs(gain_a[a_idx]),1e-30))
    RE_omega = np.mean(re_omega)
    RE_sigma = np.mean(re_sigma)
    RE_gain = np.mean(re_gain)
    RE0 = (RE_omega + RE_sigma + RE_gain) / 3.0
    delta_M_ratio = min(1.0, delta_M / M) if M>0 else 0.0
    RE = RE0 + delta_M_ratio
    return {"RE_omega":RE_omega,"RE_sigma":RE_sigma,"RE_gain":RE_gain,"RE0":RE0,
            "delta_M":delta_M,"RE":RE,"M_actual":M,"M_identified":M_tilde,"n_matched":len(matches)}

def find_pairs(exp_folder, target_folder):
    exp_path = Path(exp_folder)
    tgt_path = Path(target_folder)
    est_files = sorted(exp_path.glob("random_IR_identifiedModes_*.csv"))
    pairs = []
    for f in est_files:
        parts = f.stem.split("_")
        fid = None
        for p in reversed(parts):
            if p.isdigit():
                fid = p
                break
        if fid is None: continue
        gt = tgt_path / f"random_IR_modes_{fid}.csv"
        if gt.exists():
            pairs.append((f, gt, fid))
    return pairs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_folder", required=True)
    p.add_argument("--target_folder", required=True)
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=10000.0)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    pairs = find_pairs(args.experiment_folder, args.target_folder)
    if not pairs:
        print("No matching pairs found.")
        return
    results = []
    for est, gt, fid in pairs:
        try:
            f0_a, s_a, g_a = load_ground_truth(gt, args.fmin, args.fmax)
            f0_i, s_i, g_i = load_estimated(est)
            if args.fmin>0 or args.fmax<np.inf:
                mask = (f0_i>=args.fmin) & (f0_i<=args.fmax)
                f0_i, s_i, g_i = f0_i[mask], s_i[mask], g_i[mask]
            met = compute_metrics(f0_a, s_a, g_a, f0_i, s_i, g_i)
            results.append({"file_id":fid, **met})
            print(f"{fid}: M={met['M_actual']} M̃={met['M_identified']} ΔM={met['delta_M']} RE={met['RE']:.4f}")
        except Exception as e:
            print(f"Error {fid}: {e}")
    if results:
        df = pd.DataFrame(results)
        out_path = args.output or Path(args.experiment_folder)/"evaluation_fast.csv"
        df.to_csv(out_path, index=False)
        print("\nSummary:")
        for col in ["RE_omega","RE_sigma","RE_gain","RE0","delta_M","RE"]:
            vals = df[col].values
            print(f"{col:15s} mean={np.mean(vals):.6f} std={np.std(vals):.6f}")
        # optional plots
        plt.figure()
        plt.hist(df["RE"], bins=20)
        plt.savefig(Path(args.experiment_folder)/"RE_hist.png")
        plt.close()

if __name__ == "__main__":
    main()