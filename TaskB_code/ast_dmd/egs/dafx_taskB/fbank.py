"""Log-mel fbank front-end for AST (16 kHz)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio


@dataclass(frozen=True)
class FbankConfig:
    target_sample_rate: int = 16000
    num_mel_bins: int = 128
    target_length: int = 256
    dataset_mean: float = -4.2677393
    dataset_std: float = 4.5689974
    ir_amplitude_gain: float = 1e8


def waveform_to_fbank(waveform: torch.Tensor, sr: int, cfg: FbankConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (fbank (T, F), log_energy)."""
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform - waveform.mean()

    if sr != cfg.target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cfg.target_sample_rate)
        waveform = resampler(waveform)
        sr = cfg.target_sample_rate

    waveform = waveform * cfg.ir_amplitude_gain
    log_energy = torch.log10(torch.sum(waveform**2) + 1e-10)

    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sr,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=cfg.num_mel_bins,
        dither=0.0,
        frame_shift=10,
    )

    n_frames = fbank.shape[0]
    pad = cfg.target_length - n_frames
    if pad > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad))
    elif pad < 0:
        fbank = fbank[: cfg.target_length, :]

    fbank = (fbank - cfg.dataset_mean) / (cfg.dataset_std * 2.0)
    return fbank, log_energy
