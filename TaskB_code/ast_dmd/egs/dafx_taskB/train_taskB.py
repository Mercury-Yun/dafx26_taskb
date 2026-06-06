#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_AST = Path(__file__).resolve().parents[2]
_EGS_B = Path(__file__).resolve().parent
for p in (str(_REPO_AST), str(_EGS_B)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fbank import FbankConfig  # noqa: E402
from ast_backbone import ASTLastClsBackbone  # noqa: E402
from taskB_dataset import DAFxTaskBDataset  # noqa: E402
from taskB_grid import TaskBGrid, fit_target_norm, load_target_norm, save_target_norm  # noqa: E402
from taskB_model import AstTaskBGridModel, load_backbone_from_ckpt  # noqa: E402


# ========== Improved loss functions (compatible with existing data) ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred_logits)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


def logcosh_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = pred - target
    return torch.log(torch.cosh(err) + 1e-12)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_lambda_warmup_cosine(warmup_epochs: int, total_epochs: int):
    def fn(epoch: int) -> float:
        if warmup_epochs <= 0:
            te = max(1, total_epochs - 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, float(epoch) / float(te))))
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        t = epoch - warmup_epochs
        T = max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, float(t) / float(max(1, T - 1)))))

    return fn


def collate_batch(samples: list) -> dict:
    out: dict = {}
    for k in samples[0].keys():
        if k == "file_id":
            out[k] = [s[k] for s in samples]
        else:
            out[k] = torch.stack([s[k] for s in samples], dim=0)
    return out


def _model_inputs(batch: dict, device: torch.device, use_dmd: bool):
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


def compute_batch_loss(
    model,
    batch: dict,
    device: torch.device,
    pos_weight: torch.Tensor,
    w_sigma: float,
    w_gain: float,
    scheme_b: bool,
    w_sigma_gt: float,
    loss_type_active: str = "bce",
    loss_type_reg: str = "huber",
    freq_weight: bool = False,
    grid: TaskBGrid = None,
) -> torch.Tensor:
    use_dmd = "dmd_sigma_norm" in batch
    kwargs = _model_inputs(batch, device, use_dmd)
    al, sn, gn = model(**kwargs)
    active_t = batch["active"].to(device)

    # ----- Classification loss -----
    if loss_type_active == "bce":
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_a = bce(al, active_t)
    elif loss_type_active == "focal":
        focal = FocalLoss(alpha=pos_weight.item(), gamma=2.0)
        loss_a = focal(al, active_t)
    elif loss_type_active == "dice":
        dice = DiceLoss()
        loss_a = dice(al, active_t)
    elif loss_type_active == "bce+dice":
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        dice = DiceLoss()
        loss_a = bce(al, active_t) + dice(al, active_t)
    else:
        raise ValueError(f"Unknown loss_type_active: {loss_type_active}")

    # ----- Prepare regression targets and mask -----
    if scheme_b:
        dmd_valid = batch["dmd_valid"].to(device)
        dmd_sigma = batch["dmd_sigma_norm"].to(device)
        dmd_gain = batch["dmd_gain_norm"].to(device)
        m = (active_t > 0.5) & (dmd_valid > 0.5)
        target_s = dmd_sigma
        target_g = dmd_gain
    else:
        sigma_t = batch["sigma_norm"].to(device)
        gain_t = batch["gain_norm"].to(device)
        mask = batch["reg_mask"].to(device)
        m = mask > 0.5
        target_s = sigma_t
        target_g = gain_t

    # Frequency‑dependent weighting (lower frequencies more important)
    if freq_weight and grid is not None:
        centers = torch.from_numpy(grid.centers_hz).float().to(device)
        weight_f = torch.exp(-centers / 500.0)
        weight_f = weight_f / weight_f.mean()
    else:
        weight_f = None

    # ----- Regression loss with improved robustness -----
    if m.any():
        if loss_type_reg == "mse":
            loss_s = nn.functional.mse_loss(sn[m], target_s[m], reduction='none')
            loss_g = nn.functional.mse_loss(gn[m], target_g[m], reduction='none')
        elif loss_type_reg == "huber":
            huber = nn.HuberLoss(delta=1.0, reduction='none')
            loss_s = huber(sn[m], target_s[m])
            loss_g = huber(gn[m], target_g[m])
        elif loss_type_reg == "logcosh":
            loss_s = logcosh_loss(sn[m], target_s[m])
            loss_g = logcosh_loss(gn[m], target_g[m])
        else:
            raise ValueError(f"Unknown loss_type_reg: {loss_type_reg}")

        if weight_f is not None:
            indices = torch.where(m)[1]  # bin indices of active locations
            w = weight_f[indices]
            loss_s = loss_s * w
            loss_g = loss_g * w

        loss_s = loss_s.mean()
        loss_g = loss_g.mean()
    else:
        loss_s = torch.zeros((), device=device)
        loss_g = torch.zeros((), device=device)

    # Optional GT blend for scheme_b
    if scheme_b and w_sigma_gt > 0.0:
        sigma_t = batch["sigma_norm"].to(device)
        gain_t = batch["gain_norm"].to(device)
        m_gt = active_t > 0.5
        if m_gt.any():
            if loss_type_reg == "mse":
                loss_s_gt = nn.functional.mse_loss(sn[m_gt], sigma_t[m_gt])
                loss_g_gt = nn.functional.mse_loss(gn[m_gt], gain_t[m_gt])
            else:
                huber = nn.HuberLoss(delta=1.0)
                loss_s_gt = huber(sn[m_gt], sigma_t[m_gt])
                loss_g_gt = huber(gn[m_gt], gain_t[m_gt])
            loss_s = loss_s + w_sigma_gt * loss_s_gt
            loss_g = loss_g + w_sigma_gt * loss_g_gt

    return loss_a + w_sigma * loss_s + w_gain * loss_g


@torch.no_grad()
def evaluate_loss(
    model, loader, device, pos_weight, w_sigma, w_gain,
    scheme_b, w_sigma_gt, loss_type_active, loss_type_reg, freq_weight, grid
) -> float:
    model.eval()
    losses = []
    for batch in loader:
        loss = compute_batch_loss(
            model, batch, device, pos_weight, w_sigma, w_gain,
            scheme_b, w_sigma_gt, loss_type_active, loss_type_reg,
            freq_weight, grid
        )
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--audioset_ckpt", required=True)
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--backbone_init_ckpt", default=None, help="Optional checkpoint to warm-start AST backbone")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_backbone", action="store_true")
    p.add_argument("--target_length", type=int, default=256)
    p.add_argument("--n_bins", type=int, default=10000)
    p.add_argument("--fmin", type=float, default=20.0)
    p.add_argument("--fmax", type=float, default=10000.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone_lr", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--ir_amplitude_gain", type=float, default=1e8)
    p.add_argument("--pos_weight", type=float, default=5.0)
    p.add_argument("--w_sigma", type=float, default=3.0)
    p.add_argument("--w_gain", type=float, default=3.0)
    p.add_argument("--norm_fit_files", type=int, default=300)
    p.add_argument("--scheme_b", action="store_true")
    p.add_argument("--use_dmd_features", action="store_true")
    p.add_argument("--dmd_cache_dir", default=None)
    p.add_argument("--val_dmd_cache_dir", default=None)
    p.add_argument("--w_sigma_gt", type=float, default=0.0)

    # Improved loss options
    p.add_argument("--loss_active", type=str, default="focal",
                   choices=["bce", "focal", "dice", "bce+dice"])
    p.add_argument("--loss_reg", type=str, default="huber",
                   choices=["mse", "huber", "logcosh"])
    p.add_argument("--freq_weight", dest="freq_weight", action="store_true", default=True)
    p.add_argument("--no_freq_weight", dest="freq_weight", action="store_false",
                   help="Disable low-frequency emphasis on sigma/gain regression")

    # ----- Resume training -----
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint (e.g., last.pt or best.pt)")

    args = p.parse_args()

    if args.scheme_b:
        args.use_dmd_features = True

    freeze = not args.unfreeze_backbone
    if args.freeze_backbone:
        freeze = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    dmd_cache = args.dmd_cache_dir
    if dmd_cache is None and args.use_dmd_features:
        dmd_cache = str(exp_dir / "dmd_cache" / "train")
    val_dmd_cache = args.val_dmd_cache_dir or dmd_cache

    grid = TaskBGrid(fmin=args.fmin, fmax=args.fmax, n_bins=args.n_bins)
    norm_path = exp_dir / "target_norm.json"
    if norm_path.exists():
        target_norm = load_target_norm(norm_path)
    else:
        target_norm = fit_target_norm(Path(args.train_dir), grid, max_files=args.norm_fit_files)
        save_target_norm(norm_path, target_norm)

    fbank_cfg = FbankConfig(target_length=args.target_length, ir_amplitude_gain=float(args.ir_amplitude_gain))
    train_ds = DAFxTaskBDataset(
        args.train_dir, fbank_cfg, grid, target_norm,
        use_dmd=args.use_dmd_features, scheme_b=args.scheme_b,
        dmd_cache_dir=dmd_cache,
    )
    val_ds = DAFxTaskBDataset(
        args.val_dir, fbank_cfg, grid, target_norm,
        use_dmd=args.use_dmd_features, scheme_b=args.scheme_b,
        dmd_cache_dir=val_dmd_cache,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )

    backbone = ASTLastClsBackbone(
        input_tdim=args.target_length,
        audioset_ckpt_path=args.audioset_ckpt,
        freeze_backbone=freeze,
    )
    model = AstTaskBGridModel(
        backbone, n_bins=grid.n_bins, use_log_energy=True,
        use_dmd_features=args.use_dmd_features,
    ).to(device)

    # Load backbone from Task A checkpoint only if not resuming
    if not args.resume and args.backbone_init_ckpt:
        load_backbone_from_ckpt(model, Path(args.backbone_init_ckpt), strict=False)

    head_lr = float(args.lr)
    backbone_lr = float(args.backbone_lr) if not freeze else head_lr
    if freeze:
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=head_lr, weight_decay=args.weight_decay,
        )
    else:
        optim = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": backbone_lr},
                {"params": model.spec_enc.parameters(), "lr": head_lr},
                {"params": model.fuse.parameters(), "lr": head_lr},
                {"params": model.out_active.parameters(), "lr": head_lr},
                {"params": model.out_sigma.parameters(), "lr": head_lr},
                {"params": model.out_gain.parameters(), "lr": head_lr},
            ],
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=_lr_lambda_warmup_cosine(int(args.warmup_epochs), int(args.epochs))
    )
    pos_weight = torch.tensor([float(args.pos_weight)], device=device)

    # Resume from checkpoint if provided
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch-1}, starting from epoch {start_epoch}")
        # Update meta with possibly new epochs? We'll keep original meta but allow more epochs.
        # The meta from checkpoint may be overwritten later; we'll just keep using current args.
        # However, we might want to keep the original training history. We'll load history if exists.
        history_path = exp_dir / "history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
        else:
            history = []
    else:
        history = []

    meta = {
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "audioset_ckpt": args.audioset_ckpt,
        "ir_amplitude_gain": float(args.ir_amplitude_gain),
        "target_length": int(args.target_length),
        "grid": {"fmin": grid.fmin, "fmax": grid.fmax, "n_bins": grid.n_bins},
        "target_norm": target_norm.to_json(),
        "freeze_backbone": freeze,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": head_lr,
        "backbone_lr": backbone_lr,
        "pos_weight": args.pos_weight,
        "active_thresh": 0.5,
        "scheme_b": bool(args.scheme_b),
        "use_dmd_features": bool(args.use_dmd_features),
        "dmd_cache_dir": dmd_cache,
        "val_dmd_cache_dir": val_dmd_cache,
        "w_sigma_gt": float(args.w_sigma_gt),
        "loss_active": args.loss_active,
        "loss_reg": args.loss_reg,
        "freq_weight": args.freq_weight,
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_val = float("inf")
    # If resuming, we need to know the best_val so far. We'll load from checkpoint's best_val if stored.
    # We didn't store best_val in checkpoint previously, so we'll compute from history or just set to inf.
    # To avoid losing best model, we can load best.pt separately. Simpler: after resuming, we can evaluate
    # on validation and set best_val = current val_loss? We'll just keep best_val = inf and let it be overwritten.
    # However, we might already have a best.pt file. We'll not try to recover best_val; the new best will be saved.

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            loss = compute_batch_loss(
                model, batch, device, pos_weight,
                args.w_sigma, args.w_gain, args.scheme_b, args.w_sigma_gt,
                loss_type_active=args.loss_active,
                loss_type_reg=args.loss_reg,
                freq_weight=args.freq_weight,
                grid=grid,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(
            model, val_loader, device, pos_weight,
            args.w_sigma, args.w_gain, args.scheme_b, args.w_sigma_gt,
            args.loss_active, args.loss_reg, args.freq_weight, grid,
        )
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": time.time() - t0}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "meta": meta,
        }
        torch.save(ckpt, exp_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, exp_dir / "best.pt")

    (exp_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    train_seconds = float(sum(row.get("seconds", 0.0) for row in history))
    compute_summary = {
        "epochs_completed": len(history),
        "epochs_configured": int(args.epochs),
        "training_seconds": train_seconds,
        "best_val_loss": best_val,
    }
    (exp_dir / "compute_summary.json").write_text(
        json.dumps(compute_summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()