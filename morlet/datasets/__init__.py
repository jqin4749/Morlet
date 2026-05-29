"""SEED-family datasets for MST downstream training."""

from __future__ import annotations

import json as _json
import os
from typing import Optional

from .seed import SeedDataset, create_seed_dataset
from .seed_preprocess import (
    preprocess_seed_to_npy,
    generate_seed_splits,
    generate_seed_video_per_class_splits,
)
from .seediv import SeedIVDataset, create_seediv_dataset
from .seediv_preprocess import (
    preprocess_seediv_to_npy,
    generate_seediv_splits,
    generate_seediv_video_per_class_splits,
)
from .seedv import SeedVDataset, create_seedv_dataset
from .seedv_preprocess import (
    preprocess_seedv_to_npy,
    generate_seedv_splits,
    generate_seedv_video_per_class_splits,
)
from .seedvii import SeedVIIDataset, create_seedvii_dataset
from .seedvii_preprocess import (
    preprocess_seedvii_to_npy,
    generate_seedvii_splits,
    generate_seedvii_video_per_class_splits,
)

__all__ = [
    "SeedDataset",
    "create_seed_dataset",
    "preprocess_seed_to_npy",
    "generate_seed_splits",
    "generate_seed_video_per_class_splits",
    "SeedIVDataset",
    "create_seediv_dataset",
    "preprocess_seediv_to_npy",
    "generate_seediv_splits",
    "generate_seediv_video_per_class_splits",
    "SeedVDataset",
    "create_seedv_dataset",
    "preprocess_seedv_to_npy",
    "generate_seedv_splits",
    "generate_seedv_video_per_class_splits",
    "SeedVIIDataset",
    "create_seedvii_dataset",
    "preprocess_seedvii_to_npy",
    "generate_seedvii_splits",
    "generate_seedvii_video_per_class_splits",
    "get_dataset",
]


def _freq_baseline_n_segments_from_config(
    data_cfg: dict,
    full_args: Optional[dict] = None,
) -> int:
    if full_args:
        mc = full_args.get("model", {}).get("config") or {}
        if isinstance(mc, dict) and "freq_baseline_n_segments" in mc:
            return int(mc["freq_baseline_n_segments"])
    return int(data_cfg.get("freq_baseline_n_segments", 0))


def get_dataset(args: dict):
    """Create train and validation datasets from the configuration dict."""
    data_cfg = args["data"]
    dataset_name = data_cfg["dataset"].lower()

    if dataset_name == "seed":
        return _get_seed_dataset(data_cfg, full_args=args)
    if dataset_name == "seediv":
        return _get_seediv_dataset(data_cfg, full_args=args)
    if dataset_name in ("seedv", "seed_v"):
        return _get_seedv_dataset(data_cfg, full_args=args)
    if dataset_name in ("seedvii", "seed_vii", "seed7"):
        return _get_seedvii_dataset(data_cfg, full_args=args)

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Supported: seed, seediv, seedv, seedvii",
    )


def _get_seed_dataset(data_cfg: dict, full_args: Optional[dict] = None):
    fb = _freq_baseline_n_segments_from_config(data_cfg, full_args)
    common = dict(
        dataset_path=data_cfg["dataset_path"],
        interval_length=data_cfg.get("interval_length", 2000),
        cache_size=data_cfg.get("cache_size", 10),
        channel_last=data_cfg.get("channel_last", False),
        stride=data_cfg.get("stride"),
        padding=data_cfg.get("padding", 0),
        normalize=data_cfg.get("normalize", "window"),
        subject_stats_dir=data_cfg.get("subject_stats_dir"),
        freq_baseline_n_segments=fb,
    )

    splits_path = data_cfg.get("splits_path")
    if splits_path and os.path.isfile(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = _json.load(f)
        if "train_trial_ids" in splits:
            train_trial_ids = set(splits["train_trial_ids"])
            val_trial_ids = set(splits["val_trial_ids"])
            subjects_train = subjects_val = sessions_train = sessions_val = None
        else:
            train_trial_ids = val_trial_ids = None
            subjects_train = splits.get("train_subjects")
            subjects_val = splits.get("val_subjects")
            sessions_train = splits.get("train_sessions")
            sessions_val = splits.get("val_sessions")
    else:
        train_trial_ids = val_trial_ids = None
        subjects_train = data_cfg.get("subjects_train")
        subjects_val = data_cfg.get("subjects_val")
        sessions_train = data_cfg.get("sessions_train")
        sessions_val = data_cfg.get("sessions_val")

    train_dataset = create_seed_dataset(
        **common,
        subjects=subjects_train,
        sessions=sessions_train,
        trial_ids=train_trial_ids,
    )
    val_dataset = create_seed_dataset(
        **common,
        subjects=subjects_val,
        sessions=sessions_val,
        trial_ids=val_trial_ids,
    )
    return train_dataset, val_dataset


def _get_seediv_dataset(data_cfg: dict, full_args: Optional[dict] = None):
    fb = _freq_baseline_n_segments_from_config(data_cfg, full_args)
    common = dict(
        dataset_path=data_cfg["dataset_path"],
        interval_length=data_cfg.get("interval_length", 2000),
        cache_size=data_cfg.get("cache_size", 10),
        channel_last=data_cfg.get("channel_last", False),
        stride=data_cfg.get("stride"),
        padding=data_cfg.get("padding", 0),
        normalize=data_cfg.get("normalize", "window"),
        subject_stats_dir=data_cfg.get("subject_stats_dir"),
        freq_baseline_n_segments=fb,
    )

    splits_path = data_cfg.get("splits_path")
    if splits_path and os.path.isfile(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = _json.load(f)
        if "train_trial_ids" in splits:
            train_trial_ids = set(splits["train_trial_ids"])
            val_trial_ids = set(splits["val_trial_ids"])
            subjects_train = subjects_val = sessions_train = sessions_val = None
        else:
            train_trial_ids = val_trial_ids = None
            subjects_train = splits.get("train_subjects")
            subjects_val = splits.get("val_subjects")
            sessions_train = splits.get("train_sessions")
            sessions_val = splits.get("val_sessions")
    else:
        train_trial_ids = val_trial_ids = None
        subjects_train = data_cfg.get("subjects_train")
        subjects_val = data_cfg.get("subjects_val")
        sessions_train = data_cfg.get("sessions_train")
        sessions_val = data_cfg.get("sessions_val")

    train_dataset = create_seediv_dataset(
        **common,
        subjects=subjects_train,
        sessions=sessions_train,
        trial_ids=train_trial_ids,
    )
    val_dataset = create_seediv_dataset(
        **common,
        subjects=subjects_val,
        sessions=sessions_val,
        trial_ids=val_trial_ids,
    )
    return train_dataset, val_dataset


def _get_seedv_dataset(data_cfg: dict, full_args: Optional[dict] = None):
    fb = _freq_baseline_n_segments_from_config(data_cfg, full_args)
    common = dict(
        dataset_path=data_cfg["dataset_path"],
        interval_length=data_cfg.get("interval_length", 2000),
        cache_size=data_cfg.get("cache_size", 10),
        channel_last=data_cfg.get("channel_last", False),
        stride=data_cfg.get("stride"),
        padding=data_cfg.get("padding", 0),
        normalize=data_cfg.get("normalize", "window"),
        subject_stats_dir=data_cfg.get("subject_stats_dir"),
        freq_baseline_n_segments=fb,
    )

    splits_path = data_cfg.get("splits_path")
    if splits_path and os.path.isfile(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = _json.load(f)
        if "train_trial_ids" in splits:
            train_trial_ids = set(splits["train_trial_ids"])
            val_trial_ids = set(splits["val_trial_ids"])
            subjects_train = subjects_val = sessions_train = sessions_val = None
        else:
            train_trial_ids = val_trial_ids = None
            subjects_train = splits.get("train_subjects")
            subjects_val = splits.get("val_subjects")
            sessions_train = splits.get("train_sessions")
            sessions_val = splits.get("val_sessions")
    else:
        train_trial_ids = val_trial_ids = None
        subjects_train = data_cfg.get("subjects_train")
        subjects_val = data_cfg.get("subjects_val")
        sessions_train = data_cfg.get("sessions_train")
        sessions_val = data_cfg.get("sessions_val")

    train_dataset = create_seedv_dataset(
        **common,
        subjects=subjects_train,
        sessions=sessions_train,
        trial_ids=train_trial_ids,
    )
    val_dataset = create_seedv_dataset(
        **common,
        subjects=subjects_val,
        sessions=sessions_val,
        trial_ids=val_trial_ids,
    )
    return train_dataset, val_dataset


def _get_seedvii_dataset(data_cfg: dict, full_args: Optional[dict] = None):
    fb = _freq_baseline_n_segments_from_config(data_cfg, full_args)
    common = dict(
        dataset_path=data_cfg["dataset_path"],
        interval_length=data_cfg.get("interval_length", 2000),
        cache_size=data_cfg.get("cache_size", 10),
        channel_last=data_cfg.get("channel_last", False),
        stride=data_cfg.get("stride"),
        padding=data_cfg.get("padding", 0),
        normalize=data_cfg.get("normalize", "window"),
        subject_stats_dir=data_cfg.get("subject_stats_dir"),
        freq_baseline_n_segments=fb,
    )

    splits_path = data_cfg.get("splits_path")
    if splits_path and os.path.isfile(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = _json.load(f)
        if "train_trial_ids" in splits:
            train_trial_ids = set(splits["train_trial_ids"])
            val_trial_ids = set(splits["val_trial_ids"])
            subjects_train = subjects_val = sessions_train = sessions_val = None
        else:
            train_trial_ids = val_trial_ids = None
            subjects_train = splits.get("train_subjects")
            subjects_val = splits.get("val_subjects")
            sessions_train = splits.get("train_sessions")
            sessions_val = splits.get("val_sessions")
    else:
        train_trial_ids = val_trial_ids = None
        subjects_train = data_cfg.get("subjects_train")
        subjects_val = data_cfg.get("subjects_val")
        sessions_train = data_cfg.get("sessions_train")
        sessions_val = data_cfg.get("sessions_val")

    train_dataset = create_seedvii_dataset(
        **common,
        subjects=subjects_train,
        sessions=sessions_train,
        trial_ids=train_trial_ids,
    )
    val_dataset = create_seedvii_dataset(
        **common,
        subjects=subjects_val,
        sessions=sessions_val,
        trial_ids=val_trial_ids,
    )
    return train_dataset, val_dataset
