"""
Augmentations for MST downstream training (mostly time-domain before Morlet
wavelet tokenization; frequency-domain helpers align with the spectral front-end).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def phase_perturbation(
    x: torch.Tensor,
    sigma_phase: float,
) -> torch.Tensor:
    """Shared phase noise across channels (same epsilon per frequency bin).

    Args:
        x: [B, T, C] float32
        sigma_phase: std of Gaussian phase noise (radians)
    """
    xc = x.transpose(1, 2).contiguous()
    T = xc.shape[-1]
    spec = torch.fft.rfft(xc, dim=-1)
    amp = torch.abs(spec)
    phase = torch.angle(spec)
    B, C, F = phase.shape
    noise = torch.randn(B, F, device=x.device, dtype=x.dtype) * float(sigma_phase)
    noise = noise.unsqueeze(1).expand(B, C, F)
    phase2 = phase + noise
    real = amp * torch.cos(phase2)
    imag = amp * torch.sin(phase2)
    spec2 = torch.complex(real, imag)
    out = torch.fft.irfft(spec2, n=T, dim=-1)
    return out.transpose(1, 2).contiguous()


def _band_edges_hz(sfreq: float, n_fft: int) -> torch.Tensor:
    """Center bin frequencies for rfft (excluding Nyquist handling)."""
    n = n_fft // 2 + 1
    return torch.linspace(0.0, sfreq / 2.0, steps=n)


def random_spectral_tilt(
    amp: torch.Tensor,
    rng: np.random.Generator,
    max_slope: float = 0.3,
) -> torch.Tensor:
    """Per-sample linear gain along frequency axis in log-amplitude domain (after Morlet).

    Args:
        amp: ``[B, C, F, T]`` log-domain amplitude.
        rng: NumPy Generator (uniform slopes in ``[-max_slope, max_slope]`` per batch item).
        max_slope: half-width of the uniform slope distribution.
    """
    f_dim = amp.shape[2]
    slope = torch.from_numpy(
        rng.uniform(-max_slope, max_slope, size=(amp.shape[0], 1, 1, 1)).astype(
            np.float32
        )
    ).to(device=amp.device, dtype=amp.dtype)
    freq_axis = torch.linspace(
        -1, 1, f_dim, device=amp.device, dtype=amp.dtype
    ).view(1, 1, f_dim, 1)
    return amp + slope * freq_axis


def random_channel_gain(
    x: torch.Tensor,
    rng: np.random.Generator,
    log_std: float = 0.3,
) -> torch.Tensor:
    """Per-sample, per-channel multiplicative gain on raw EEG (before wavelet).

    Args:
        x: ``[B, T, C]``.
        log_std: std of Normal noise on log-gain (gain = exp(noise)).
    """
    b, _, c = x.shape
    log_gain = torch.from_numpy(
        rng.normal(0, log_std, size=(b, 1, c)).astype(np.float32)
    ).to(device=x.device, dtype=x.dtype)
    return x * torch.exp(log_gain)


def band_specific_noise(
    x: torch.Tensor,
    sfreq: float,
    alpha: float,
    n_bands: int,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Per-channel band-limited noise scaled to local signal power in each band.

    x: [B, T, C]
    """
    if rng is None:
        rng = np.random.default_rng()
    B, T, C = x.shape
    device, dtype = x.device, x.dtype
    x_c = x.transpose(1, 2).contiguous()
    spec = torch.fft.rfft(x_c, dim=-1)
    freqs = _band_edges_hz(sfreq, T).to(device=device, dtype=dtype)
    n_bins = spec.shape[-1]

    lo_hz = 2.0
    hi_hz = min(45.0, sfreq / 2.0 - 1.0)
    edges = np.geomspace(lo_hz, hi_hz, num=n_bands + 1)
    out = x_c.clone()
    for b in range(n_bands):
        f0, f1 = edges[b], edges[b + 1]
        mask = ((freqs >= f0) & (freqs < f1)).float()
        if mask.sum() < 1:
            continue
        for c in range(C):
            n_time = torch.randn(B, T, device=device, dtype=dtype)
            n_spec = torch.fft.rfft(n_time, dim=-1)
            n_spec = n_spec * mask.unsqueeze(0)
            n_t = torch.fft.irfft(n_spec, n=T, dim=-1)
            sig = x_c[:, c, :]
            bp = torch.fft.irfft(torch.fft.rfft(sig, dim=-1) * mask.view(1, -1), n=T, dim=-1)
            p_sig = (bp**2).mean(dim=-1, keepdim=True).clamp_min(1e-10)
            p_n = (n_t**2).mean(dim=-1, keepdim=True).clamp_min(1e-10)
            scale = float(alpha) * torch.sqrt(p_sig / p_n)
            out[:, c, :] = out[:, c, :] + n_t * scale
    return out.transpose(1, 2).contiguous()


def apply_intralabel_random_interpolation(
    eeg: torch.Tensor,
    labels: Optional[torch.Tensor],
    dataset_ids: Optional[torch.Tensor],
    *,
    enabled: bool,
    apply_prob: float,
    alpha_min: float,
    alpha_max: float,
) -> torch.Tensor:
    """IntraLabelMix: interpolate samples with same (dataset_id, label) in-batch.

    For each sample i in a (dataset_id, label) group, with probability ``apply_prob``:
        x_i <- a * x_i + (1 - a) * x_j
    where j is randomly chosen from the same (dataset_id, label) group and
    a ~ Uniform(alpha_min, alpha_max).

    ``eeg`` must be batch-first: ``[B, ...]`` (e.g. ``[B, T, C]`` EEG or
    ``[B, C, F, T]`` wavelet log-amplitude).
    """
    if (not enabled) or labels is None or dataset_ids is None:
        return eeg
    if eeg.ndim < 2 or eeg.shape[0] < 2:
        return eeg
    if apply_prob <= 0:
        return eeg

    alpha_lo = min(float(alpha_min), float(alpha_max))
    alpha_hi = max(float(alpha_min), float(alpha_max))
    labels_list = labels.detach().cpu().tolist()
    dsids_list = dataset_ids.detach().cpu().tolist()

    group_to_indices: Dict[tuple, List[int]] = {}
    for i, (y, d) in enumerate(zip(labels_list, dsids_list)):
        yi = int(y)
        di = int(d)
        if yi < 0:
            continue
        if di < 0:
            continue
        group_to_indices.setdefault((di, yi), []).append(i)

    mixed = eeg.clone()
    for idxs in group_to_indices.values():
        if len(idxs) < 2:
            continue
        for i in idxs:
            if random.random() >= apply_prob:
                continue
            partner_candidates = [j for j in idxs if j != i]
            if not partner_candidates:
                continue
            j = random.choice(partner_candidates)
            alpha = random.uniform(alpha_lo, alpha_hi)
            mixed[i] = alpha * eeg[i] + (1.0 - alpha) * eeg[j]

    return mixed


def apply_wavelet_time_roll(
    tf: torch.Tensor,
    *,
    rng: Optional[np.random.Generator],
    enabled: bool,
    apply_prob: float,
) -> torch.Tensor:
    """Circular shift along the last dimension (pooled wavelet time axis).

    For each batch row independently, with probability ``apply_prob``, roll by a
    random offset in ``{1, ..., T-1}`` (no-op offset 0 is excluded when ``T > 1``).

    Args:
        tf: ``[B, ..., T]`` (e.g. ``[B, C, F, T]`` log-amplitude).
        rng: NumPy Generator; if None or not ``enabled``, returns ``tf`` unchanged.
    """
    if (not enabled) or rng is None or apply_prob <= 0:
        return tf
    if tf.ndim < 2:
        return tf
    t_len = int(tf.shape[-1])
    if t_len <= 1:
        return tf
    b = int(tf.shape[0])
    out = []
    for i in range(b):
        row = tf[i]
        if float(rng.random()) >= float(apply_prob):
            out.append(row)
            continue
        shift = int(rng.integers(1, t_len))
        out.append(torch.roll(row, shifts=shift, dims=-1))
    return torch.stack(out, dim=0)


def apply_intralabel_random_interpolation_frequency(
    eeg: torch.Tensor,
    labels: Optional[torch.Tensor],
    dataset_ids: Optional[torch.Tensor],
    *,
    enabled: bool,
    apply_prob: float,
    alpha_min: float,
    alpha_max: float,
) -> torch.Tensor:
    """IntraLabelMix along rFFT bins (time axis), same grouping as time-domain mix.

    For each sample *i* in a *(dataset_id, label)* group, with probability
    ``apply_prob``:

        S_i <- a * S_i + (1 - a) * S_j

    where *S* is ``torch.fft.rfft`` of each channel (last dim = time), then
    ``irfft`` back to ``[B, T, C]``.

    For real-valued EEG, linear mixing in the complex spectrum equals mixing
    the time signal: ``irfft(a S_i + (1-a) S_j) = a x_i + (1-a) x_j``. The
    frequency-domain formulation matches spectrum models that emphasize
    spectral structure; numerically it can differ slightly from direct time
    mixing due to FFT rounding.

    Args:
        eeg: ``[B, T, C]`` real-valued.
    """
    if (not enabled) or labels is None or dataset_ids is None:
        return eeg
    if eeg.ndim != 3 or eeg.shape[0] < 2:
        return eeg
    if apply_prob <= 0:
        return eeg

    alpha_lo = min(float(alpha_min), float(alpha_max))
    alpha_hi = max(float(alpha_min), float(alpha_max))
    labels_list = labels.detach().cpu().tolist()
    dsids_list = dataset_ids.detach().cpu().tolist()

    group_to_indices: Dict[tuple, List[int]] = {}
    for i, (y, d) in enumerate(zip(labels_list, dsids_list)):
        yi = int(y)
        di = int(d)
        if yi < 0:
            continue
        if di < 0:
            continue
        group_to_indices.setdefault((di, yi), []).append(i)

    _, T, _ = eeg.shape
    xc = eeg.transpose(1, 2).contiguous()
    spec = torch.fft.rfft(xc, dim=-1)
    spec_mixed = spec.clone()

    for idxs in group_to_indices.values():
        if len(idxs) < 2:
            continue
        for i in idxs:
            if random.random() >= apply_prob:
                continue
            partner_candidates = [j for j in idxs if j != i]
            if not partner_candidates:
                continue
            j = random.choice(partner_candidates)
            alpha = random.uniform(alpha_lo, alpha_hi)
            spec_mixed[i] = alpha * spec[i] + (1.0 - alpha) * spec[j]

    out_c = torch.fft.irfft(spec_mixed, n=T, dim=-1)
    return out_c.transpose(1, 2).contiguous()


def random_channel_dropout_mask(
    B: int,
    C: int,
    p_drop: float,
    device: torch.device,
    dtype=torch.bool,
) -> torch.Tensor:
    """Bernoulli per entry; True = dropped."""
    return torch.rand(B, C, device=device) < float(p_drop)


def apply_augmentation_view(
    x: torch.Tensor,
    *,
    sfreq: float = 200.0,
    rng: Optional[np.random.Generator] = None,
    probs: Optional[Dict[str, float]] = None,
    channel_gain_log_std: float = 0.3,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    One stochastic view. Returns (x_aug, channel_dropout_mask or None).
    channel_dropout_mask: [B, C] bool, True = dropped (handled in tokenizer).

    Keys in *probs* (all optional except noted):
      - phase, band_noise, channel_dropout: Bernoulli probabilities for applying
        each augmentation (defaults 0.5 / 0.5 / 0.3).
      - channel_gain: Bernoulli probability for per-channel log-normal gain (default 0 = off).
      - channel_drop_fraction: if set, per-channel drop probability when channel
        dropout runs; if unset, sample uniformly in [0.1, 0.3] (legacy default).
    """
    if rng is None:
        rng = np.random.default_rng()
    p = probs or {}
    p_phase = float(p.get("phase", 0.5))
    p_band = float(p.get("band_noise", 0.5))
    p_ch = float(p.get("channel_dropout", 0.3))
    p_cgain = float(p.get("channel_gain", 0.0))

    B, T, C = x.shape
    device = x.device
    dtype = x.dtype
    x = x.clone()

    did_cgain = p_cgain > 0.0 and rng.random() < p_cgain
    did_phase = rng.random() < p_phase
    did_band = rng.random() < p_band
    did_ch = rng.random() < p_ch

    if did_cgain:
        x = random_channel_gain(x, rng, log_std=float(channel_gain_log_std))

    if did_phase:
        sig = float(rng.uniform(math.pi / 6.0, math.pi / 2.0))
        x = phase_perturbation(x, sig)

    if did_band:
        K = int(rng.integers(1, 4))
        al = float(rng.uniform(0.1, 0.3))
        x = band_specific_noise(x, sfreq, al, K, rng=rng)

    ch_mask: Optional[torch.Tensor] = None
    if did_ch:
        if p.get("channel_drop_fraction") is not None:
            p_drop = float(p["channel_drop_fraction"])
            p_drop = max(0.0, min(1.0, p_drop))
        else:
            p_drop = float(rng.uniform(0.1, 0.3))
        ch_mask = random_channel_dropout_mask(B, C, p_drop, device, dtype=torch.bool)

    if not (did_cgain or did_phase or did_band or did_ch):
        sig = float(rng.uniform(math.pi / 6.0, math.pi / 3.0))
        x = phase_perturbation(x, sig)

    return x, ch_mask


def apply_augmentation_view_downstream(
    x: torch.Tensor,
    *,
    sfreq: float = 200.0,
    rng: Optional[np.random.Generator] = None,
    probs: Optional[Dict[str, float]] = None,
    labels: Optional[torch.Tensor] = None,
    dataset_ids: Optional[torch.Tensor] = None,
    intralabel_enabled: bool = False,
    intralabel_apply_prob: float = 0.0,
    intralabel_alpha_min: float = 0.2,
    intralabel_alpha_max: float = 0.8,
    intralabel_frequency_domain: bool = False,
    channel_gain_log_std: float = 0.3,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """MST downstream pipeline: optional IntraLabelMix, then ``apply_augmentation_view``."""
    mix_fn = (
        apply_intralabel_random_interpolation_frequency
        if intralabel_frequency_domain
        else apply_intralabel_random_interpolation
    )
    x_in = mix_fn(
        x,
        labels,
        dataset_ids,
        enabled=intralabel_enabled,
        apply_prob=intralabel_apply_prob,
        alpha_min=intralabel_alpha_min,
        alpha_max=intralabel_alpha_max,
    )
    return apply_augmentation_view(
        x_in,
        sfreq=sfreq,
        rng=rng,
        probs=probs,
        channel_gain_log_std=channel_gain_log_std,
    )
