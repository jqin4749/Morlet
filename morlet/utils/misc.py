"""Lightweight training helpers (subset of the upstream EEG toolkit)."""

from __future__ import annotations

import torch


def gpu_mem_info(quiet=False):
    t = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
    r = torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024
    a = torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024
    f = t - a
    if not quiet:
        print(
            f"GPU memory: total {t:.2f}GB, free {f:.2f}GB, allocated {a:.2f}GB, reserved {r:.2f}GB"
        )
    return t, f, a, r


def skip_logic(name, skip_list):
    for skip_name in skip_list:
        if skip_name in name:
            return True
    return False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or skip_logic(name, skip_list):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def calculate_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    trainable_params = 0
    non_trainable_params = 0
    total_size_bytes = 0

    for p in model.parameters():
        n = p.numel()
        total_size_bytes += n * p.element_size()
        if p.requires_grad:
            trainable_params += n
        else:
            non_trainable_params += n

    for b in model.buffers():
        total_size_bytes += b.numel() * b.element_size()

    total_params = trainable_params + non_trainable_params

    def _fmt_params(n):
        if n >= 1e9:
            return f"{n / 1e9:.2f} B"
        if n >= 1e6:
            return f"{n / 1e6:.2f} M"
        if n >= 1e3:
            return f"{n / 1e3:.2f} K"
        return str(n)

    def _fmt_size(size_bytes):
        if size_bytes >= 1 << 30:
            return f"{size_bytes / (1 << 30):.2f} GB"
        if size_bytes >= 1 << 20:
            return f"{size_bytes / (1 << 20):.2f} MB"
        if size_bytes >= 1 << 10:
            return f"{size_bytes / (1 << 10):.2f} KB"
        return f"{size_bytes} B"

    print(f"Total parameters:         {_fmt_params(total_params)} ({total_params:,})")
    print(f"  Trainable:              {_fmt_params(trainable_params)} ({trainable_params:,})")
    print(f"  Non-trainable (frozen): {_fmt_params(non_trainable_params)} ({non_trainable_params:,})")
    print(f"Estimated storage size:   {_fmt_size(total_size_bytes)}")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "size_bytes": total_size_bytes,
    }
