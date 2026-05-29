"""MST (Morlet Spectral Transformer) model factory."""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from .mst import MSTDownstreamConfig, MSTDownstreamModel, MSTPretrainConfig
from .types import ModelOutput

__all__ = [
    "MODEL_REGISTRY",
    "get_model",
    "MSTDownstreamConfig",
    "MSTDownstreamModel",
    "MSTPretrainConfig",
    "ModelOutput",
]


MODEL_REGISTRY = {
    "mst_downstream": (MSTDownstreamModel, MSTDownstreamConfig),
}


def get_model(args: Dict[str, Any], return_loaded_keys: bool = False):
    """Instantiate the model from the configuration dict.

    Supports optional loading from a pretrained checkpoint (``model.pretrained_path``).
    """
    model_cfg = args["model"]
    name = model_cfg["name"].lower()

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )

    ModelClass, ConfigClass = MODEL_REGISTRY[name]
    config = ConfigClass(**model_cfg.get("config", {}))
    model = ModelClass(config)

    loaded_keys = set()
    pretrained_path = model_cfg.get("pretrained_path")
    if pretrained_path and os.path.isdir(pretrained_path):
        print(f"[get_model] Loading pretrained weights from {pretrained_path}")
        safetensors_path = os.path.join(pretrained_path, "model.safetensors")
        pytorch_path = os.path.join(pretrained_path, "pytorch_model.bin")
        if os.path.isfile(safetensors_path):
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
        elif os.path.isfile(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(f"No model weights found in {pretrained_path}")
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("._cached_B")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        loaded_keys = set(model.state_dict().keys()) & set(state_dict.keys())
    elif pretrained_path and os.path.isfile(pretrained_path):
        print(f"[get_model] Loading pretrained state_dict from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("._cached_B")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")
        loaded_keys = set(model.state_dict().keys()) & set(state_dict.keys())

    if return_loaded_keys:
        return (model, loaded_keys)
    return model
