"""AST backbone: last-layer CLS embedding for Task B."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_AST = Path(__file__).resolve().parents[2]
if str(_REPO_AST) not in sys.path:
    sys.path.insert(0, str(_REPO_AST))

from src.models.ast_models import ASTModel


def adapt_backbone_checkpoint_state_dict(state_dict: dict) -> dict:
    """Rename legacy head keys so external checkpoints load into the backbone."""
    adapted: dict = {}
    for key, tensor in state_dict.items():
        if key.startswith("head.body."):
            adapted["head.net." + key[len("head.body.") :]] = tensor
        else:
            adapted[key] = tensor
    return adapted


class ASTLastClsBackbone(nn.Module):
    """Audio Spectrogram Transformer with CLS token output."""

    def __init__(
        self,
        input_tdim: int,
        audioset_ckpt_path: str,
        fstride: int = 10,
        tstride: int = 10,
        model_size: str = "base384",
        freeze_backbone: bool = True,
    ):
        super().__init__()
        ckpt_src = Path(audioset_ckpt_path).expanduser().resolve()
        ckpt_dst = _REPO_AST / "pretrained_models" / "audioset_10_10_0.4593.pth"
        ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
        if not (ckpt_dst.exists() or ckpt_dst.is_symlink()):
            if not ckpt_src.exists():
                raise FileNotFoundError(f"AudioSet checkpoint not found: {ckpt_src}")
            try:
                os.symlink(str(ckpt_src), str(ckpt_dst))
            except (FileExistsError, OSError):
                shutil.copy2(str(ckpt_src), str(ckpt_dst))

        base = ASTModel(
            label_dim=527,
            fstride=fstride,
            tstride=tstride,
            input_fdim=128,
            input_tdim=input_tdim,
            imagenet_pretrain=True,
            audioset_pretrain=True,
            model_size=model_size,
            verbose=False,
        )
        self.v = base.v
        self.embed_dim = int(self.v.pos_embed.shape[-1])

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).transpose(2, 3)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        return x[:, 0]
