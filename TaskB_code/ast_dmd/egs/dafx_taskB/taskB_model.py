from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_AST = Path(__file__).resolve().parents[2]
_EGS_B = Path(__file__).resolve().parent
for p in (str(_REPO_AST), str(_EGS_B)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ast_backbone import ASTLastClsBackbone  # noqa: E402


class SpectrumEncoder1D(nn.Module):
    """Encode log-magnitude curve (+ optional DMD channels) on the frequency grid."""

    def __init__(self, n_bins: int, in_channels: int = 1, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, hidden, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
        )
        self.proj = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, n_bins)
        x = self.net(x)
        x = self.proj(x)
        return x


class AstTaskBGridModel(nn.Module):
    """
    AST CLS (+ log energy) fused with 1D spectrum features -> per-bin
    active logit, sigma_norm, gain_norm.

    use_dmd_features: concat [band_mag, dmd_sigma_norm, dmd_gain_norm, dmd_valid] as 4 channels.
    """

    def __init__(
        self,
        backbone: ASTLastClsBackbone,
        n_bins: int,
        use_log_energy: bool = True,
        use_dmd_features: bool = False,
        hidden: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.use_log_energy = bool(use_log_energy)
        self.use_dmd_features = bool(use_dmd_features)
        self.n_bins = int(n_bins)
        in_ch = 4 if use_dmd_features else 1
        self.spec_enc = SpectrumEncoder1D(n_bins, in_channels=in_ch, hidden=hidden)
        global_dim = backbone.embed_dim + (1 if use_log_energy else 0)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden + global_dim, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.out_active = nn.Conv1d(hidden, 1, kernel_size=1)
        self.out_sigma = nn.Conv1d(hidden, 1, kernel_size=1)
        self.out_gain = nn.Conv1d(hidden, 1, kernel_size=1)

    def _align_spec(self, spec: torch.Tensor, target_len: int) -> torch.Tensor:
        if spec.shape[-1] == target_len:
            return spec
        return torch.nn.functional.interpolate(spec, size=target_len, mode="linear", align_corners=False)

    def _build_spec_input(
        self,
        band_mag: torch.Tensor,
        dmd_sigma_norm: torch.Tensor | None = None,
        dmd_gain_norm: torch.Tensor | None = None,
        dmd_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = band_mag.unsqueeze(1)
        if not self.use_dmd_features:
            return x
        if dmd_sigma_norm is None or dmd_gain_norm is None or dmd_valid is None:
            raise ValueError("DMD tensors required when use_dmd_features=True")
        return torch.stack(
            [band_mag, dmd_sigma_norm, dmd_gain_norm, dmd_valid],
            dim=1,
        )

    def forward(
        self,
        fbank: torch.Tensor,
        band_mag: torch.Tensor,
        log_energy: torch.Tensor | None = None,
        dmd_sigma_norm: torch.Tensor | None = None,
        dmd_gain_norm: torch.Tensor | None = None,
        dmd_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls = self.backbone(fbank)
        if self.use_log_energy:
            if log_energy is None:
                raise ValueError("log_energy required")
            le = log_energy.view(-1, 1).to(device=cls.device, dtype=cls.dtype)
            g = torch.cat([cls, le], dim=-1)
        else:
            g = cls

        spec_in = self._build_spec_input(band_mag, dmd_sigma_norm, dmd_gain_norm, dmd_valid)
        spec = self.spec_enc(spec_in)
        spec = self._align_spec(spec, self.n_bins)
        g_map = g.unsqueeze(-1).expand(-1, -1, self.n_bins)
        h = self.fuse(torch.cat([spec, g_map], dim=1))
        active_logit = self.out_active(h).squeeze(1)
        sigma = self.out_sigma(h).squeeze(1)
        gain = self.out_gain(h).squeeze(1)
        return active_logit, sigma, gain


def load_backbone_from_ckpt(model: AstTaskBGridModel, ckpt_path: Path, strict: bool = False) -> None:
    from ast_backbone import adapt_backbone_checkpoint_state_dict

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    sd = adapt_backbone_checkpoint_state_dict(ckpt["model_state"])
    bb_sd = {k[len("backbone.") :]: v for k, v in sd.items() if k.startswith("backbone.")}
    model.backbone.load_state_dict(bb_sd, strict=strict)
