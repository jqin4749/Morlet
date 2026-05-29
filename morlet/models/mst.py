"""
Morlet Spectral Transformer (MST) — downstream classification model.

Morlet wavelet tokenization, frequency-specific spatial projection, and a 2D RoPE Transformer
backbone with a dual-pooling classification head (concatenated CLS and mean-pooled patches).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from ..utils.augmentation import (
    apply_intralabel_random_interpolation,
    apply_wavelet_time_roll,
    random_spectral_tilt,
)

from .types import ModelOutput


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class MSTPretrainConfig(PretrainedConfig):
    model_type = "mst_pretrain"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Wavelet
        self.n_freqs = kwargs.get("n_freqs", 20)
        self.freq_low = kwargs.get("freq_low", 2.0)
        self.freq_high = kwargs.get("freq_high", 45.0)
        self.wavelet_cycles = kwargs.get("wavelet_cycles", 5.0)
        self.sfreq = kwargs.get("sfreq", 200.0)
        # Time pooling → n_time_frames bins (default 8 ≈ 250 ms each at 2 s / 200 Hz)
        self.n_time_frames = kwargs.get("n_time_frames", 8)
        self.log_eps = kwargs.get("log_eps", 1e-6)
        self.morlet_kernel_std = kwargs.get("morlet_kernel_std", 7.0)
        # Spatial
        self.n_channels = kwargs.get("n_channels", 62)
        self.spatial_dim = kwargs.get("spatial_dim", 16)
        # Transformer
        self.embed_dim = kwargs.get("embed_dim", 256)
        self.num_layers = kwargs.get("num_layers", 8)
        self.num_heads = kwargs.get("num_heads", 8)
        self.dropout = kwargs.get("dropout", 0.1)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        # SupCon
        self.supcon_proj_dim = kwargs.get("supcon_proj_dim", 128)
        self.supcon_tau = kwargs.get("supcon_tau", 0.1)
        self.num_datasets = kwargs.get("num_datasets", 8)
        self.dataset_num_classes = kwargs.get("dataset_num_classes", [3, 4, 5, 7, 6, 0, 0, 0])
        # VICReg
        self.vicreg_weight = kwargs.get("vicreg_weight", 1.0)
        self.vicreg_lambda_inv = kwargs.get("vicreg_lambda_inv", 25.0)
        self.vicreg_lambda_var = kwargs.get("vicreg_lambda_var", 25.0)
        self.vicreg_lambda_cov = kwargs.get("vicreg_lambda_cov", 1.0)
        self.vicreg_var_gamma = kwargs.get("vicreg_var_gamma", 1.0)
        # Backward-compatibility: this field may appear in older configs/checkpoints,
        # but the current implementation does not use uncertainty-weighted SupCon.
        self.uncertainty_init = kwargs.get("uncertainty_init", 1.0)
        # Optional axial attention (even layers: time axis, odd: freq) for ablations
        self.use_axial_attention = kwargs.get("use_axial_attention", False)
        # Wavelet-domain baseline: neighboring-segment mean subtracted; when >0, Morlet output is
        # ``cat([raw_log, raw_log - baseline], dim=channel)`` → ``2 * n_channels`` wavelet channels.
        self.freq_baseline_n_segments = int(kwargs.get("freq_baseline_n_segments", 0))


class MSTDownstreamConfig(MSTPretrainConfig):
    """Same backbone hyperparameters as the shared config; adds classification head size."""

    model_type = "mst_downstream"

    def __init__(self, **kwargs):
        self.downstream_output_dim = int(kwargs.pop("downstream_output_dim", 3))
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# Morlet wavelet (fixed, complex conv1d)
# ---------------------------------------------------------------------------


def _make_complex_morlet_kernels(
    sfreq: float,
    center_freqs_hz: np.ndarray,
    n_cycles: float,
    kernel_std_mult: float = 7.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked real/imag kernels [F, 1, L] numpy float32."""
    kernels_r = []
    kernels_i = []
    for fc in center_freqs_hz:
        fc = max(float(fc), 1e-3)
        sigma_s = float(n_cycles) / (2.0 * math.pi * fc)
        half_width_s = kernel_std_mult * sigma_s
        half_len = int(max(3, math.ceil(half_width_s * sfreq)))
        n = 2 * half_len + 1
        t = (np.arange(n, dtype=np.float64) - half_len) / sfreq
        omega = 2.0 * math.pi * fc
        env = np.exp(-(t**2) / (2.0 * sigma_s**2))
        wave = env * np.exp(1j * omega * t)
        wave = wave.astype(np.complex128)
        wave /= np.sqrt(np.sum(np.abs(wave) ** 2) + 1e-12)
        kernels_r.append(np.real(wave).astype(np.float32).reshape(1, -1))
        kernels_i.append(np.imag(wave).astype(np.float32).reshape(1, -1))
    max_len = max(k.shape[1] for k in kernels_r)
    out_r = np.zeros((len(kernels_r), 1, max_len), dtype=np.float32)
    out_i = np.zeros((len(kernels_i), 1, max_len), dtype=np.float32)
    for idx, (kr, ki) in enumerate(zip(kernels_r, kernels_i)):
        L = kr.shape[1]
        pad_l = (max_len - L) // 2
        out_r[idx, 0, pad_l : pad_l + L] = kr[0]
        out_i[idx, 0, pad_l : pad_l + L] = ki[0]
    return out_r, out_i


class MorletWaveletTransform(nn.Module):
    """Fixed complex Morlet wavelets per frequency; |conv| then adaptive time pool + log."""

    def __init__(
        self,
        sfreq: float,
        n_freqs: int,
        freq_low: float,
        freq_high: float,
        n_cycles: float,
        n_time_frames: int,
        log_eps: float,
        kernel_std_mult: float = 7.0,
        freq_baseline_n_segments: int = 0,
    ):
        super().__init__()
        freqs = np.geomspace(freq_low, freq_high, num=n_freqs).astype(np.float64)
        kr, ki = _make_complex_morlet_kernels(sfreq, freqs, n_cycles, kernel_std_mult)
        self.register_buffer("kernel_real", torch.from_numpy(kr), persistent=False)
        self.register_buffer("kernel_imag", torch.from_numpy(ki), persistent=False)
        self.n_freqs = n_freqs
        self.n_time_frames = n_time_frames
        self.log_eps = log_eps
        self.kernel_size = kr.shape[-1]
        self.padding = self.kernel_size // 2
        self.freq_baseline_n_segments = int(freq_baseline_n_segments)

    def _forward_log_amplitude(
        self, x: torch.Tensor, n_time_frames: int,
    ) -> torch.Tensor:
        """x: [B, C, T] -> [B, C, F, n_time_frames] log |Morlet| (after pool)."""
        B, C, T = x.shape
        f_n = self.kernel_real.shape[0]
        x_flat = x.reshape(B * C, 1, T)
        kr = self.kernel_real.to(dtype=x.dtype, device=x.device)
        ki = self.kernel_imag.to(dtype=x.dtype, device=x.device)
        yr = F.conv1d(x_flat, kr, padding=self.padding)
        yi = F.conv1d(x_flat, ki, padding=self.padding)
        if yr.shape[-1] != T:
            if yr.shape[-1] > T:
                yr = yr[..., :T]
                yi = yi[..., :T]
            else:
                pad = T - yr.shape[-1]
                yr = F.pad(yr, (0, pad))
                yi = F.pad(yi, (0, pad))
        amp = torch.sqrt(yr**2 + yi**2 + 1e-12)
        amp = amp.reshape(B, C, f_n, T)
        b_f, c_f, ff, tt = amp.shape
        amp = amp.reshape(b_f * c_f * ff, 1, tt)
        amp = F.adaptive_avg_pool1d(amp, n_time_frames)
        amp = amp.reshape(B, C, f_n, n_time_frames)
        return torch.log(amp + self.log_eps)

    def forward(
        self,
        x: torch.Tensor,
        n_time_frames: Optional[int] = None,
        x_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] real
            n_time_frames: optional override for adaptive time pooling bins (e.g. DINO local crops)
            x_history: optional [B, H, C, T] same layout as x, where
                ``H = 2 * freq_baseline_n_segments`` (previous n + next n, excluding self);
                if unset and
                ``freq_baseline_n_segments > 0``, the residual branch is raw log (same as first branch).
        Returns:
            [B, C, F, T_frames] if ``freq_baseline_n_segments == 0``; else
            [B, 2*C, F, T_frames] with ``[:, :C]`` raw log amp and ``[:, C:]`` raw minus mean prior log amp.
        """
        n_tf = int(n_time_frames) if n_time_frames is not None else self.n_time_frames
        cur = self._forward_log_amplitude(x, n_tf)
        if self.freq_baseline_n_segments <= 0:
            return cur
        residual = cur
        if x_history is not None:
            bh, hh, ch, th = x_history.shape
            expected_h = 2 * self.freq_baseline_n_segments
            if hh != expected_h:
                raise ValueError(
                    f"x_history has H={hh} but expected H={expected_h} "
                    f"(2 * freq_baseline_n_segments, n previous + n next)",
                )
            if ch != x.shape[1] or th != x.shape[2]:
                raise ValueError(
                    f"x_history shape {(bh, hh, ch, th)} incompatible with x "
                    f"{(x.shape[0], x.shape[1], x.shape[2])}",
                )
            flat = x_history.reshape(bh * hh, ch, th)
            hist = self._forward_log_amplitude(flat, n_tf)
            hist = hist.view(bh, hh, ch, self.n_freqs, n_tf)
            baseline = hist.mean(dim=1)
            residual = cur - baseline
        return torch.cat([cur, residual], dim=1)

# ---------------------------------------------------------------------------
# Spatial projection & mask token
# ---------------------------------------------------------------------------


class LearnableMaskToken(nn.Module):
    def __init__(self, n_freqs: int):
        super().__init__()
        self.mask_values = nn.Parameter(torch.zeros(n_freqs))

    def forward(self) -> torch.Tensor:
        return self.mask_values


class FrequencySpecificSpatialProjection(nn.Module):
    """W_c^f: [F, C, D] applied per (b,f,t) slice."""

    def __init__(self, n_freqs: int, n_channels: int, spatial_dim: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_channels = n_channels
        self.spatial_dim = spatial_dim
        w = torch.empty(n_freqs, n_channels, spatial_dim)
        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.projections = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, F, T] -> [B, D, F, T]
        return torch.einsum("bcft,fcd->bdft", x, self.projections)


class MSTTokenizer(nn.Module):
    def __init__(self, config: MSTPretrainConfig, mask_module: LearnableMaskToken):
        super().__init__()
        self.freq_baseline_n_segments = int(
            getattr(config, "freq_baseline_n_segments", 0),
        )
        self.wavelet = MorletWaveletTransform(
            sfreq=config.sfreq,
            n_freqs=config.n_freqs,
            freq_low=config.freq_low,
            freq_high=config.freq_high,
            n_cycles=config.wavelet_cycles,
            n_time_frames=config.n_time_frames,
            log_eps=config.log_eps,
            kernel_std_mult=config.morlet_kernel_std,
            freq_baseline_n_segments=self.freq_baseline_n_segments,
        )

        self.mask_module = mask_module
        n_wav_ch = (
            config.n_channels * 2
            if self.freq_baseline_n_segments > 0
            else config.n_channels
        )
        self.spatial = FrequencySpecificSpatialProjection(
            config.n_freqs, n_wav_ch, config.spatial_dim
        )
        self.proj = nn.Linear(config.spatial_dim, config.embed_dim)
        self.n_freqs = config.n_freqs
        self.n_time_frames = config.n_time_frames
        self.n_channels = config.n_channels

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
        bad_channel_mask: Optional[torch.Tensor] = None,
        n_time_frames: Optional[int] = None,
        eeg_history: Optional[torch.Tensor] = None,
        spectral_tilt_rng: Optional[np.random.Generator] = None,
        spectral_tilt_prob: float = 0.0,
        spectral_tilt_max_slope: float = 0.3,
        labels: Optional[torch.Tensor] = None,
        dataset_ids: Optional[torch.Tensor] = None,
        post_wavelet_intralabel_enabled: bool = False,
        post_wavelet_intralabel_apply_prob: float = 0.0,
        post_wavelet_intralabel_alpha_min: float = 0.2,
        post_wavelet_intralabel_alpha_max: float = 0.8,
        post_wavelet_time_roll_enabled: bool = False,
        post_wavelet_time_roll_apply_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] EEG
            channel_mask: [B, C] bool, True = dropped (apply mask token after wavelet)
            bad_channel_mask: [B, C] bool, True = bad/padded channel (same filling)
            n_time_frames: optional override passed to Morlet adaptive pool (DINO multi-crop)
            eeg_history: optional [B, H, T, C] neighboring segments
                (H = ``2 * freq_baseline_n_segments``, n previous + n next, excluding self)
            spectral_tilt_rng: with ``spectral_tilt_prob`` > 0 and ``self.training``, may apply
                log-amplitude frequency tilt after Morlet (before channel mask fill).
            labels / dataset_ids: optional [B] for post-wavelet IntraLabel mix (train only).
            post_wavelet_*: optional augmentations on Morlet log-amplitude (train only).
        Returns:
            [B, F*T, embed_dim] patch tokens (no CLS). When freq baseline is on, Morlet uses
            ``2 * n_channels`` wavelet channels (raw log concat with baseline-removed log).
        """
        x = x.transpose(1, 2).contiguous()
        xh = None
        if (
            eeg_history is not None
            and self.freq_baseline_n_segments > 0
        ):
            xh = eeg_history.transpose(2, 3).contiguous()
        tf = self.wavelet(x, n_time_frames=n_time_frames, x_history=xh)
        if self.training:
            tf = apply_intralabel_random_interpolation(
                tf,
                labels,
                dataset_ids,
                enabled=bool(post_wavelet_intralabel_enabled),
                apply_prob=float(post_wavelet_intralabel_apply_prob),
                alpha_min=float(post_wavelet_intralabel_alpha_min),
                alpha_max=float(post_wavelet_intralabel_alpha_max),
            )
            tf = apply_wavelet_time_roll(
                tf,
                rng=spectral_tilt_rng,
                enabled=bool(post_wavelet_time_roll_enabled),
                apply_prob=float(post_wavelet_time_roll_apply_prob),
            )
        if (
            self.training
            and spectral_tilt_rng is not None
            and float(spectral_tilt_prob) > 0.0
            and spectral_tilt_rng.random() < float(spectral_tilt_prob)
        ):
            tf = random_spectral_tilt(
                tf,
                spectral_tilt_rng,
                max_slope=float(spectral_tilt_max_slope),
            )
        mv = self.mask_module.mask_values.view(1, 1, self.n_freqs, 1).to(dtype=tf.dtype)
        mask_drop = channel_mask if channel_mask is not None else None
        mask_bad = bad_channel_mask if bad_channel_mask is not None else None
        if mask_drop is not None or mask_bad is not None:
            combined = torch.zeros(
                x.shape[0], self.n_channels, dtype=torch.bool, device=x.device
            )
            if mask_drop is not None:
                combined = combined | mask_drop.bool()
            if mask_bad is not None:
                combined = combined | mask_bad.bool()
            if combined.any():
                if self.freq_baseline_n_segments > 0:
                    combined = torch.cat([combined, combined], dim=1)
                tf = torch.where(combined[:, :, None, None], mv, tf)
        h = self.spatial(tf)
        B, D, Freq, Tfr = h.shape
        h = h.permute(0, 2, 3, 1).contiguous().reshape(B, Freq * Tfr, D)
        return self.proj(h)


# ---------------------------------------------------------------------------
# 2D RoPE: first half of dim = time, second half = frequency
# ---------------------------------------------------------------------------


def _apply_rope_interleaved(
    x: torch.Tensor,
    pos_ids: torch.Tensor,
    inv_freq: torch.Tensor,
) -> torch.Tensor:
    """x: [B, H, S, D], D even; pos_ids: [S] long; inv_freq: [D//2]."""
    freqs = pos_ids.float().view(1, 1, -1, 1) * inv_freq.view(1, 1, 1, -1).to(
        dtype=x.dtype, device=x.device
    )
    cos = freqs.cos().to(dtype=x.dtype)
    sin = freqs.sin().to(dtype=x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = y1
    out[..., 1::2] = y2
    return out


def apply_2d_rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    pos_time: torch.Tensor,
    pos_freq: torch.Tensor,
    inv_freq_time: torch.Tensor,
    inv_freq_freq: torch.Tensor,
    apply_rope_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, H, S, Dh]. First Dh//2 dims: time RoPE; last Dh//2: freq RoPE.
    apply_rope_mask: [S] bool, True = apply RoPE (False for CLS).
    """
    B, H, S, Dh = q.shape
    if Dh < 4:
        return q, k
    h1 = Dh // 2
    h2 = Dh - h1
    qt, qf = q[..., :h1], q[..., h1:]
    kt, kf = k[..., :h1], k[..., h1:]
    inv_t = inv_freq_time.to(device=q.device, dtype=q.dtype)
    inv_f = inv_freq_freq.to(device=q.device, dtype=q.dtype)
    qt2 = _apply_rope_interleaved(qt, pos_time, inv_t)
    kt2 = _apply_rope_interleaved(kt, pos_time, inv_t)
    qf2 = _apply_rope_interleaved(qf, pos_freq, inv_f)
    kf2 = _apply_rope_interleaved(kf, pos_freq, inv_f)
    q_out = torch.cat([qt2, qf2], dim=-1)
    k_out = torch.cat([kt2, kf2], dim=-1)
    if apply_rope_mask is not None:
        m = apply_rope_mask.view(1, 1, S, 1).to(dtype=q.dtype)
        q_out = q * (1 - m) + q_out * m
        k_out = k * (1 - m) + k_out * m
    return q_out, k_out


class RotaryEmbedding2D(nn.Module):
    """Inverse frequencies for time / freq halves of each head dimension."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        h1 = head_dim // 2
        h2 = head_dim - h1
        inv_t = 1.0 / (
            base ** (torch.arange(0, h1, 2, dtype=torch.float32) / float(h1))
        )
        inv_f = 1.0 / (
            base ** (torch.arange(0, h2, 2, dtype=torch.float32) / float(h2))
        )
        self.register_buffer("inv_freq_time", inv_t, persistent=False)
        self.register_buffer("inv_freq_freq", inv_f, persistent=False)


# ---------------------------------------------------------------------------
# Transformer block (full attention + pre-LN + optional axial)
# ---------------------------------------------------------------------------


class MSTTransformerBlock(nn.Module):
    def __init__(self, config: MSTPretrainConfig, rope: RotaryEmbedding2D, layer_idx: int):
        super().__init__()
        D = config.embed_dim
        H = config.num_heads
        assert D % H == 0
        self.dim = D
        self.num_heads = H
        self.head_dim = D // H
        self.dropout_p = config.dropout
        self.layer_idx = layer_idx
        self.use_axial = config.use_axial_attention
        self.n_freqs = config.n_freqs
        self.n_time = config.n_time_frames

        self.norm1 = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.Linear(D, D)

        self.norm2 = nn.LayerNorm(D)
        hidden = int(D * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, D),
            nn.Dropout(config.dropout),
        )
        self.rope = rope

    def _full_attention(
        self,
        x: torch.Tensor,
        pos_time: torch.Tensor,
        pos_freq: torch.Tensor,
        cls_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        q = self.q_proj(x).view(B, S, H, Dh).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, S, H, Dh).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, S, H, Dh).permute(0, 2, 1, 3)
        q, k = apply_2d_rope_qk(
            q,
            k,
            pos_time,
            pos_freq,
            self.rope.inv_freq_time,
            self.rope.inv_freq_freq,
            cls_mask,
        )
        ctx = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        ctx = ctx.transpose(1, 2).reshape(B, S, H * Dh)
        return self.out_proj(ctx)

    def _axial_attention(
        self,
        x: torch.Tensor,
        pos_time: torch.Tensor,
        pos_freq: torch.Tensor,
        cls_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Even layer: time axis within each frequency row; odd: freq within each time column."""
        B, S, D = x.shape
        Freq, Tn = self.n_freqs, self.n_time
        assert S == 1 + Freq * Tn
        patches = x[:, 1:, :].reshape(B, Freq, Tn, D)
        cls_tok = x[:, :1, :]
        H, Dh = self.num_heads, self.head_dim
        time_axis = self.layer_idx % 2 == 0
        rope_on = torch.ones(1, dtype=torch.bool, device=x.device)

        def attn_1d(seq: torch.Tensor, pt: torch.Tensor, pf: torch.Tensor) -> torch.Tensor:
            Bb, L, _ = seq.shape
            q = self.q_proj(seq).view(Bb, L, H, Dh).permute(0, 2, 1, 3)
            k = self.k_proj(seq).view(Bb, L, H, Dh).permute(0, 2, 1, 3)
            v = self.v_proj(seq).view(Bb, L, H, Dh).permute(0, 2, 1, 3)
            q, k = apply_2d_rope_qk(
                q,
                k,
                pt,
                pf,
                self.rope.inv_freq_time,
                self.rope.inv_freq_freq,
                rope_on.expand(L),
            )
            ctx = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0
            )
            ctx = ctx.transpose(1, 2).reshape(Bb, L, H * Dh)
            return self.out_proj(ctx)

        pt_all = pos_time[1:]
        pf_all = pos_freq[1:]
        if time_axis:
            rows = []
            for f in range(Freq):
                row = patches[:, f, :, :]
                sl = f * Tn + torch.arange(Tn, device=x.device)
                rows.append(attn_1d(row, pt_all[sl].long(), pf_all[sl].long()))
            out_seq = torch.stack(rows, dim=1).reshape(B, Freq * Tn, D)
        else:
            cols = []
            for t in range(Tn):
                col = patches[:, :, t, :]
                sl = torch.arange(Freq, device=x.device) * Tn + t
                cols.append(attn_1d(col, pt_all[sl].long(), pf_all[sl].long()))
            out_seq = torch.stack(cols, dim=2).reshape(B, Freq * Tn, D)

        cls_out = self._full_attention(x, pos_time, pos_freq, cls_mask)[:, :1, :]
        return torch.cat([cls_out, out_seq], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        pos_time: torch.Tensor,
        pos_freq: torch.Tensor,
        cls_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        if self.use_axial:
            attn_out = self._axial_attention(x_norm, pos_time, pos_freq, cls_mask)
        else:
            attn_out = self._full_attention(x_norm, pos_time, pos_freq, cls_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class MSTDownstreamModel(PreTrainedModel):
    """MST backbone + linear head for downstream classification.

    Optional ``pretrained_path`` in training config can load compatible weights
    (e.g. from an MST pretrain checkpoint) with ``strict=False``; the classification
    head is always newly initialized.
    """

    config_class = MSTDownstreamConfig

    def __init__(self, config: MSTDownstreamConfig):
        super().__init__(config)
        self.config = config
        self.tokenizer = MSTTokenizer(config, LearnableMaskToken(config.n_freqs))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.rope = RotaryEmbedding2D(config.embed_dim // config.num_heads)
        self.blocks = nn.ModuleList(
            [
                MSTTransformerBlock(config, self.rope, layer_idx=i)
                for i in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.downstream_head = nn.Linear(
            2 * config.embed_dim, config.downstream_output_dim
        )
        n_patches = config.n_freqs * config.n_time_frames
        pos_t = torch.zeros(1 + n_patches, dtype=torch.long)
        pos_f = torch.zeros(1 + n_patches, dtype=torch.long)
        idx = 0
        for f in range(config.n_freqs):
            for t in range(config.n_time_frames):
                idx += 1
                pos_t[idx] = t + 1
                pos_f[idx] = f + 1
        self.register_buffer("pos_time", pos_t, persistent=False)
        self.register_buffer("pos_freq", pos_f, persistent=False)
        rope_mask = torch.ones(1 + n_patches, dtype=torch.bool)
        rope_mask[0] = False
        self.register_buffer("rope_apply_mask", rope_mask, persistent=False)

        self.post_init()

    def forward(
        self,
        x: torch.Tensor,
        bad_channel_mask: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        augmentation_rng: Optional[np.random.Generator] = None,
        spectral_tilt_prob: float = 0.0,
        spectral_tilt_max_slope: float = 0.3,
        eeg_history: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        dataset_ids: Optional[torch.Tensor] = None,
        post_wavelet_intralabel_enabled: bool = False,
        post_wavelet_intralabel_apply_prob: float = 0.0,
        post_wavelet_intralabel_alpha_min: float = 0.2,
        post_wavelet_intralabel_alpha_max: float = 0.8,
        post_wavelet_time_roll_enabled: bool = False,
        post_wavelet_time_roll_apply_prob: float = 0.0,
    ) -> ModelOutput:
        """
        Args:
            x: [B, T, C]
            bad_channel_mask / channel_mask: [B, C] bool, optional
            augmentation_rng: optional NumPy RNG for train-time spectral tilt (log-domain).
            spectral_tilt_prob: Bernoulli probability to apply tilt when rng is set.
            eeg_history: optional [B, H, T, C] for wavelet-domain segment baseline.
            labels / dataset_ids: optional [B] for post-wavelet IntraLabel (train only).
            post_wavelet_*: optional Morlet-domain augmentations (train only).
        Returns:
            predictions: [B, num_classes]
            last_hidden_state: concat(z_cls, mean(patch)) [B, 2D]
            hidden_states: patch tokens [B, F*T, D]
        """
        patches = self.tokenizer(
            x,
            channel_mask,
            bad_channel_mask,
            eeg_history=eeg_history,
            spectral_tilt_rng=augmentation_rng,
            spectral_tilt_prob=spectral_tilt_prob,
            spectral_tilt_max_slope=spectral_tilt_max_slope,
            labels=labels,
            dataset_ids=dataset_ids,
            post_wavelet_intralabel_enabled=post_wavelet_intralabel_enabled,
            post_wavelet_intralabel_apply_prob=post_wavelet_intralabel_apply_prob,
            post_wavelet_intralabel_alpha_min=post_wavelet_intralabel_alpha_min,
            post_wavelet_intralabel_alpha_max=post_wavelet_intralabel_alpha_max,
            post_wavelet_time_roll_enabled=post_wavelet_time_roll_enabled,
            post_wavelet_time_roll_apply_prob=post_wavelet_time_roll_apply_prob,
        )
        B = patches.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls, patches], dim=1)
        pos_t = self.pos_time.to(h.device)
        pos_f = self.pos_freq.to(h.device)
        cls_m = self.rope_apply_mask.to(h.device)
        for blk in self.blocks:
            h = blk(h, pos_t, pos_f, cls_m)
        h = self.norm(h)
        z_cls = h[:, 0]
        z_pool = h[:, 1:].mean(dim=1)
        pooled_rep = torch.cat([z_cls, z_pool], dim=-1)
        logits = self.downstream_head(pooled_rep)
        patch_states = h[:, 1:, :]
        return ModelOutput(
            predictions=logits,
            last_hidden_state=pooled_rep,
            hidden_states=patch_states,
        )
