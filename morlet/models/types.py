"""Shared model output dataclass (compatible with Hugging Face-style ``forward`` returns)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutput:
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    proto_loss: Optional[torch.Tensor] = None
    z_cls: Optional[torch.Tensor] = None
