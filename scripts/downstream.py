"""
Downstream emotion classification training for MST (Morlet Spectral Transformer).

Usage:
    pip install -e .
    python scripts/downstream.py --config configs/mst_seed_subject_independent.json

    # Multi-GPU
    accelerate launch scripts/downstream.py --config configs/mst_seed_subject_independent.json
"""

import os
import json
import math
import argparse
import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from typing import Optional, Dict, List
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
from datetime import timedelta
from diffusers.optimization import get_scheduler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
)

from morlet.models import get_model
from morlet.datasets import get_dataset
from morlet.utils.augmentation import apply_augmentation_view_downstream
from morlet.utils.misc import add_weight_decay, gpu_mem_info, model_summary


def _mst_eeg_history_kw(batch: dict) -> dict:
    """Keyword args for MST when batch includes prior segments for baseline removal."""
    eh = batch.get("eeg_history")
    if eh is None:
        return {}
    return {"eeg_history": eh.float()}


# ============================================================================
#  Classification metrics (matching LaBraM evaluation)
# ============================================================================
_HIGHER_IS_BETTER = {
    "accuracy", "balanced_accuracy", "pr_auc", "roc_auc",
    "cohen_kappa", "f1_weighted", "f1_micro", "f1_macro",
}


def compute_classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    metrics_list: list,
    is_binary: bool,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics with sklearn (LaBraM-compatible, no pyhealth)."""
    out = {}
    preds = preds.float()
    if is_binary:
        pred_proba = torch.sigmoid(preds).numpy().ravel()
        targets_np = targets.numpy().ravel().astype(np.int64)
        pred_labels = (pred_proba >= threshold).astype(np.int64)
        n_pos = int(targets_np.sum())
        n_neg = len(targets_np) - n_pos
        has_both = n_pos > 0 and n_neg > 0

        if "accuracy" in metrics_list:
            out["accuracy"] = float(accuracy_score(targets_np, pred_labels))
        if "balanced_accuracy" in metrics_list:
            out["balanced_accuracy"] = float(balanced_accuracy_score(targets_np, pred_labels))
        if "roc_auc" in metrics_list:
            if has_both:
                out["roc_auc"] = float(roc_auc_score(targets_np, pred_proba))
            else:
                out["roc_auc"] = 0.0
        if "pr_auc" in metrics_list:
            if has_both:
                out["pr_auc"] = float(average_precision_score(targets_np, pred_proba))
            else:
                out["pr_auc"] = 0.0
        return out
    else:
        pred_labels = preds.numpy()
        if pred_labels.ndim > 1:
            pred_labels = np.argmax(pred_labels, axis=-1)
        targets_np = targets.numpy().ravel().astype(np.int64)

        if "accuracy" in metrics_list:
            out["accuracy"] = float(accuracy_score(targets_np, pred_labels))
        if "balanced_accuracy" in metrics_list:
            out["balanced_accuracy"] = float(balanced_accuracy_score(targets_np, pred_labels))
        if "cohen_kappa" in metrics_list:
            out["cohen_kappa"] = float(cohen_kappa_score(targets_np, pred_labels))
        if "f1_weighted" in metrics_list:
            out["f1_weighted"] = float(f1_score(targets_np, pred_labels, average="weighted", zero_division=0))
        return out


class FocalBCELoss(nn.Module):
    """Binary focal loss on logits with optional alpha balancing."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits, targets: [B, 1] or [B]
        logits = logits.view(-1)
        targets = targets.view(-1).float()
        # Sigmoid probabilities
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        return (focal_weight * bce).mean()


# ============================================================================
#  get_loss_fn
# ============================================================================
def get_loss_fn(args: dict):
    """Return a loss function based on configuration."""
    loss_name = args["training"].get("loss", "mse").lower()
    loss_cfg = args["training"].get("loss_config", {})
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    elif loss_name == "cross_entropy":
        class_weights = loss_cfg.get("class_weights")
        if class_weights is not None and class_weights != []:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weight)
        return nn.CrossEntropyLoss()
    elif loss_name == "smooth_cross_entropy":
        class_weights = loss_cfg.get("class_weights")
        if class_weights is not None and class_weights != []:
            weight = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    elif loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "focal_bce":
        alpha = float(loss_cfg.get("alpha", 0.25))
        gamma = float(loss_cfg.get("gamma", 2.0))
        return FocalBCELoss(alpha=alpha, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


# ============================================================================
#  Target extraction from batch
# ============================================================================
def extract_target(batch: dict, data_cfg: dict, train_cfg: dict) -> torch.Tensor:
    """Extract and transform the target tensor from a data batch.

    For classification tasks (``train_cfg["task_type"] == "classification"``),
    returns the ``"label"`` field.  Binary targets are shaped ``[B, 1]`` float;
    multiclass targets are ``[B]`` long.

    For regression tasks, returns the temporal target reduced along time.
    """
    task_type = train_cfg.get("task_type", "regression")

    if task_type == "classification":
        target = batch["label"]
        is_binary = train_cfg.get("is_binary", False)
        if is_binary:
            return target.float().unsqueeze(-1)   # [B, 1] for BCEWithLogitsLoss
        return target.long()                       # [B]   for CrossEntropyLoss

    # ---- regression path (unchanged) ----
    target_key = data_cfg.get("target", "joints")
    target_transform = data_cfg.get("target_transform", "mean")

    target = batch[target_key]  # e.g. [B, T, D] for joints

    if target.dim() == 3:
        if target_transform == "mean":
            target = target.mean(dim=1)          # [B, D]
        elif target_transform == "last":
            target = target[:, -1, :]            # [B, D]
        elif target_transform == "first":
            target = target[:, 0, :]             # [B, D]
        else:
            raise ValueError(f"Unknown target_transform: {target_transform}")

    return target.float()


def safe_collate_batch(batch):
    """Collate a list of sample dicts without shared-storage resize pitfalls.

    Some environments (notably with newer Python/PyTorch combinations) can hit
    ``RuntimeError: Trying to resize storage that is not resizable`` inside the
    default multi-worker collate path. This collate stacks tensor fields
    explicitly and falls back to ``default_collate`` for non-tensor fields.
    """
    if not batch:
        return batch

    elem = batch[0]
    if isinstance(elem, dict):
        out = {}
        for k in elem.keys():
            vals = [sample[k] for sample in batch]
            if isinstance(vals[0], torch.Tensor):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = default_collate(vals)
        return out

    return default_collate(batch)


# ============================================================================
#  Validation
# ============================================================================
@torch.no_grad()
def validate(
    model,
    val_loader,
    loss_fn,
    data_cfg,
    train_cfg,
    accelerator,
    global_step,
    current_lr=None,
    model_cfg=None,
):
    """Run a full validation pass and log metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_proto_loss = 0.0
    proto_loss_count = 0

    all_preds = []
    all_targets = []

    task_type = train_cfg.get("task_type", "regression")
    is_binary = train_cfg.get("is_binary", False)
    pass_bad_channel_mask = (
        model_cfg is not None
        and model_cfg.get("name") == "mst_downstream"
    )

    for batch in val_loader:
        eeg = batch["eeg"].float()
        target = extract_target(batch, data_cfg, train_cfg)
        mst_hist = {}
        if model_cfg is not None and model_cfg.get("name") == "mst_downstream":
            mst_hist = _mst_eeg_history_kw(batch)

        if pass_bad_channel_mask:
            output = model(
                eeg,
                bad_channel_mask=batch.get("bad_channel_mask"),
                **mst_hist,
            )
        else:
            output = model(eeg, **mst_hist) if mst_hist else model(eeg)
        preds = output.predictions

        if task_type != "classification" and preds.shape != target.shape:
            target = target[..., :preds.shape[-1]]

        loss = loss_fn(preds, target)
        bz = eeg.shape[0]
        total_loss += loss.item() * bz
        total_samples += bz
        if getattr(output, "proto_loss", None) is not None:
            total_proto_loss += output.proto_loss.item() * bz
            proto_loss_count += bz

        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

    avg_loss = total_loss / max(total_samples, 1)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = {"val_loss": avg_loss}
    if current_lr is not None:
        metrics["val_lr"] = float(current_lr)
    if proto_loss_count > 0:
        metrics["val_proto_loss"] = total_proto_loss / proto_loss_count

    if task_type == "classification":
        metrics_list = train_cfg.get(
            "metrics",
            ["accuracy", "balanced_accuracy"]
            if not is_binary
            else ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
        )
        if is_binary:
            preds_for_metric = all_preds.squeeze(-1)
            targets_for_metric = all_targets.squeeze(-1)
        else:
            preds_for_metric = all_preds
            targets_for_metric = all_targets
        cls_metrics = compute_classification_metrics(
            preds_for_metric, targets_for_metric, metrics_list, is_binary,
        )
        for k, v in cls_metrics.items():
            metrics[f"val_{k}"] = float(v)
    else:
        if all_targets.dim() >= 1 and all_targets.shape[-1] > 0:
            ss_res = ((all_targets - all_preds) ** 2).sum()
            ss_tot = ((all_targets - all_targets.mean(dim=0, keepdim=True)) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            metrics["val_r2"] = r2.item()

        # Optional: plot ground truth vs prediction for regression tasks
        if accelerator.is_main_process:
            # Flatten to 1D for plotting; support [N, 1] or [N, D]
            gt = all_targets.detach().cpu().numpy()
            pr = all_preds.detach().cpu().numpy()
            if gt.ndim > 1:
                gt = gt.reshape(gt.shape[0], -1)[:, 0]
            if pr.ndim > 1:
                pr = pr.reshape(pr.shape[0], -1)[:, 0]

            # For very large N, just plot a prefix to keep the figure readable
            max_points = 512
            if gt.shape[0] > max_points:
                gt_plot = gt[:max_points]
                pr_plot = pr[:max_points]
                x = np.arange(max_points)
            else:
                gt_plot = gt
                pr_plot = pr
                x = np.arange(gt.shape[0])

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, gt_plot, label="Ground truth", linewidth=1.5)
            ax.plot(x, pr_plot, label="Prediction", linewidth=1.0, alpha=0.8)
            ax.set_xlabel("Validation sample index")
            ax.set_ylabel("Target value")
            ax.set_title("Regression: ground truth vs prediction")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.2)

            accelerator.log({"val_gt_pred_plot": fig}, step=global_step)
            plt.close(fig)

    if accelerator.is_main_process:
        accelerator.log(metrics, step=global_step)

    return metrics


# ============================================================================
#  Latent quality analysis
# ============================================================================

_DS_COLORS = ["#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0",
               "#00BCD4", "#F44336", "#8BC34A", "#3F51B5", "#FF5722"]


def _few_shot_trial_disjoint_knn(
    embeds: np.ndarray,
    labels: np.ndarray,
    trials: np.ndarray,
    *,
    k: int,
    n_shot: int,
    n_episodes: int,
    min_trials_per_class: int,
    seed: int,
) -> Dict[str, float]:
    from sklearn.neighbors import KNeighborsClassifier

    class_to_trial_to_indices: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx, (lbl, tr) in enumerate(zip(labels.tolist(), trials.tolist())):
        class_to_trial_to_indices[int(lbl)][int(tr)].append(idx)

    eligible_classes = [
        cls for cls, trial_map in class_to_trial_to_indices.items()
        if len(trial_map) >= max(min_trials_per_class, n_shot + 1)
    ]
    if not eligible_classes:
        return {
            "knn_accuracy": float("nan"),
            "knn_acc_std": float("nan"),
            "knn_valid_episodes": 0.0,
            "knn_leak_violations": 0.0,
            "knn_trial_field_present": 1.0,
            "knn_trial_unique_count": float(len(np.unique(trials))),
        }

    rng = np.random.default_rng(seed)
    n_neighbors = max(1, int(k))
    episode_accs: List[float] = []
    leak_violations = 0

    for _ in range(max(1, n_episodes)):
        support_idx: List[int] = []
        query_idx: List[int] = []

        for cls in eligible_classes:
            trial_keys = list(class_to_trial_to_indices[cls].keys())
            if len(trial_keys) < n_shot + 1:
                support_idx = []
                query_idx = []
                break

            chosen_support = rng.choice(len(trial_keys), size=n_shot, replace=False)
            support_trials = {trial_keys[int(i)] for i in chosen_support.tolist()}
            query_trials = set(trial_keys) - support_trials
            if not query_trials:
                support_idx = []
                query_idx = []
                break
            if support_trials.intersection(query_trials):
                leak_violations += 1
                support_idx = []
                query_idx = []
                break

            for tr in support_trials:
                support_idx.extend(class_to_trial_to_indices[cls][tr])
            for tr in query_trials:
                query_idx.extend(class_to_trial_to_indices[cls][tr])

        if not support_idx or not query_idx:
            continue

        Xs = embeds[support_idx]
        ys = labels[support_idx]
        Xq = embeds[query_idx]
        yq = labels[query_idx]
        if Xs.shape[0] < n_neighbors:
            continue

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        knn.fit(Xs, ys)
        episode_accs.append(float(knn.score(Xq, yq)))

    if not episode_accs:
        return {
            "knn_accuracy": float("nan"),
            "knn_acc_std": float("nan"),
            "knn_valid_episodes": 0.0,
            "knn_leak_violations": float(leak_violations),
            "knn_trial_field_present": 1.0,
            "knn_trial_unique_count": float(len(np.unique(trials))),
        }

    return {
        "knn_accuracy": float(np.mean(episode_accs)),
        "knn_acc_std": float(np.std(episode_accs)),
        "knn_valid_episodes": float(len(episode_accs)),
        "knn_leak_violations": float(leak_violations),
        "knn_trial_field_present": 1.0,
        "knn_trial_unique_count": float(len(np.unique(trials))),
    }


def _evaluate_latent_clustering(
    latent_np: np.ndarray,   # [N, D]
    labels_arr: np.ndarray,  # [N]
    trial_arr: Optional[np.ndarray] = None,  # [N]
    n_components: int = 10,
    knn_eval_cfg: Optional[dict] = None,
) -> dict:
    """Supervised clustering quality metrics in PCA space."""
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                                 normalized_mutual_info_score)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    cfg = knn_eval_cfg or {}
    protocol = str(cfg.get("protocol", "few_shot_trial_disjoint")).strip().lower()
    require_trial_for_fewshot = bool(cfg.get("require_trial_for_fewshot", True))
    few_shot_cfg = cfg.get("few_shot", {})

    metrics: dict = {}
    N, D = latent_np.shape
    n_classes = len(np.unique(labels_arr))
    n_comp = min(n_components, D, N - 1)

    pca = PCA(n_components=n_comp)
    emb = pca.fit_transform(latent_np)
    metrics["pca_explained_variance"] = float(np.sum(pca.explained_variance_ratio_))

    if N > n_classes:
        metrics["silhouette"] = float(silhouette_score(emb, labels_arr))

    km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    km_labels = km.fit_predict(emb)
    metrics["ARI"] = float(adjusted_rand_score(labels_arr, km_labels))
    metrics["NMI"] = float(normalized_mutual_info_score(labels_arr, km_labels))

    if protocol == "few_shot_trial_disjoint":
        if trial_arr is None:
            if require_trial_for_fewshot:
                raise ValueError(
                    "[downstream] Missing trial ids for few_shot_trial_disjoint KNN evaluation."
                )
            metrics["knn_accuracy"] = float("nan")
        else:
            if np.any(trial_arr < 0):
                raise ValueError(
                    "[downstream] Invalid trial ids (<0) found while running few-shot trial-disjoint KNN."
                )
            fs_metrics = _few_shot_trial_disjoint_knn(
                emb,
                labels_arr,
                trial_arr,
                k=int(few_shot_cfg.get("k", 5)),
                n_shot=int(few_shot_cfg.get("n_shot", 5)),
                n_episodes=int(few_shot_cfg.get("n_episodes", 100)),
                min_trials_per_class=int(few_shot_cfg.get("min_trials_per_class", 6)),
                seed=int(few_shot_cfg.get("seed", 42)),
            )
            metrics.update(fs_metrics)
    else:
        n_splits = min(5, N // max(n_classes, 1))
        if n_splits >= 2:
            knn = KNeighborsClassifier(n_neighbors=5)
            cv_scores = cross_val_score(knn, emb, labels_arr, cv=n_splits)
            metrics["knn_accuracy"] = float(np.mean(cv_scores))

    return metrics


def _plot_latent_pca(
    latent_np: np.ndarray,            # [N, D]
    labels: np.ndarray,               # [N]
    global_step: int,
    label_names: Optional[dict] = None,
) -> dict:
    """2D and 3D PCA scatter plots for pooled [N, D] representations."""
    from sklearn.decomposition import PCA

    unique_labels = np.unique(labels)
    if label_names is None:
        label_names = {l: str(l) for l in unique_labels}
    color_map = {l: _DS_COLORS[i % len(_DS_COLORS)] for i, l in enumerate(unique_labels)}

    figs: dict = {}

    pca2 = PCA(n_components=2)
    emb2 = pca2.fit_transform(latent_np)
    fig2, ax = plt.subplots(figsize=(8, 6))
    for l in unique_labels:
        mask = labels == l
        ax.scatter(emb2[mask, 0], emb2[mask, 1],
                   c=[color_map[l]], label=label_names.get(int(l), str(l)),
                   alpha=0.6, s=15, linewidths=0)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"Latent PCA 2D — step {global_step}")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    figs["pca_2d"] = fig2

    if latent_np.shape[1] >= 3:
        pca3 = PCA(n_components=3)
        emb3 = pca3.fit_transform(latent_np)
        fig3 = plt.figure(figsize=(9, 7))
        ax3 = fig3.add_subplot(111, projection="3d")
        for l in unique_labels:
            mask = labels == l
            ax3.scatter(emb3[mask, 0], emb3[mask, 1], emb3[mask, 2],
                        c=[color_map[l]], label=label_names.get(int(l), str(l)),
                        alpha=0.6, s=12, linewidths=0)
        ax3.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0] * 100:.1f}%)")
        ax3.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1] * 100:.1f}%)")
        ax3.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2] * 100:.1f}%)")
        ax3.set_title(f"Latent PCA 3D — step {global_step}")
        ax3.legend(fontsize=8)
        fig3.tight_layout()
        figs["pca_3d"] = fig3

    return figs


def _single_trajectory_metrics(traj: np.ndarray) -> dict:
    """Compute geometry/kinematics metrics for a single discrete trajectory.

    Args:
        traj: [T, D] — one latent trajectory in any-dimensional space.

    Returns:
        dict with keys: mean_speed, std_speed, tortuosity,
        mean_turning_angle, std_turning_angle, radius_of_gyration.
    """
    T, D = traj.shape
    metrics: dict = {}

    velocities = traj[1:] - traj[:-1]           # [T-1, D]
    speeds = np.linalg.norm(velocities, axis=1)  # [T-1]
    speeds_safe = np.where(speeds == 0, 1e-8, speeds)

    metrics["mean_speed"] = float(np.mean(speeds))
    metrics["std_speed"] = float(np.std(speeds))

    path_length = float(np.sum(speeds))
    end_to_end = float(np.linalg.norm(traj[-1] - traj[0]))
    metrics["tortuosity"] = path_length / (end_to_end + 1e-8)

    if T > 2:
        v1 = velocities[:-1]    # [T-2, D]
        v2 = velocities[1:]     # [T-2, D]
        dot = np.sum(v1 * v2, axis=1)
        mags = speeds_safe[:-1] * speeds_safe[1:]
        cos_a = np.clip(dot / mags, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(cos_a))
        metrics["mean_turning_angle"] = float(np.mean(angles_deg))
        metrics["std_turning_angle"] = float(np.std(angles_deg))
    else:
        metrics["mean_turning_angle"] = 0.0
        metrics["std_turning_angle"] = 0.0

    centroid = traj.mean(axis=0)
    metrics["radius_of_gyration"] = float(
        np.sqrt(np.mean(np.sum((traj - centroid) ** 2, axis=1)))
    )
    return metrics


def _compute_trajectory_geometry(
    latents_np: np.ndarray,   # [N, T, D]
    labels_arr: np.ndarray,   # [N]
) -> dict:
    """Per-class trajectory geometry metrics computed in the raw latent space.

    For each sample, calls _single_trajectory_metrics on its [T, D] trajectory
    (no PCA — raw high-dimensional space is more faithful).  Returns per-class
    mean and std for each metric, keyed as ``traj/{metric}_{mean|std}_class{c}``.
    """
    N = latents_np.shape[0]
    unique_labels = np.unique(labels_arr)

    per_sample = [_single_trajectory_metrics(latents_np[i]) for i in range(N)]
    metric_keys = list(per_sample[0].keys())

    result: dict = {}
    for lbl in unique_labels:
        mask = labels_arr == lbl
        for k in metric_keys:
            vals = np.array([per_sample[i][k] for i in range(N) if mask[i]])
            result[f"traj/{k}_mean_class{lbl}"] = float(np.mean(vals))
            result[f"traj/{k}_std_class{lbl}"] = float(np.std(vals))
    return result


def _plot_trajectory_geometry_boxplots(
    latents_np: np.ndarray,          # [N, T, D]
    labels_arr: np.ndarray,          # [N]
    global_step: int,
    label_names: Optional[dict] = None,
):
    """One-way ANOVA and per-class boxplots for trajectory geometry metrics.

    Returns:
        anova_dict  — scalar dict keyed ``traj/anova_{fstat|pvalue}_{metric}``
        fig         — matplotlib Figure (2×3 grid, one subplot per metric)
    """
    from scipy.stats import f_oneway

    N = latents_np.shape[0]
    unique_labels = np.unique(labels_arr)
    if label_names is None:
        label_names = {l: str(l) for l in unique_labels}

    per_sample = [_single_trajectory_metrics(latents_np[i]) for i in range(N)]
    metric_keys = list(per_sample[0].keys())

    # Group raw values per class
    per_class_vals: Dict[str, Dict] = {k: {} for k in metric_keys}
    for k in metric_keys:
        for lbl in unique_labels:
            mask = labels_arr == lbl
            per_class_vals[k][lbl] = [per_sample[i][k] for i in range(N) if mask[i]]

    # One-way ANOVA
    anova_dict: dict = {}
    for k in metric_keys:
        groups = [per_class_vals[k][lbl] for lbl in unique_labels
                  if len(per_class_vals[k][lbl]) >= 2]
        if len(groups) >= 2:
            fstat, pval = f_oneway(*groups)
            anova_dict[f"traj/anova_fstat_{k}"] = float(fstat)
            anova_dict[f"traj/anova_pvalue_{k}"] = float(pval)

    # Boxplot grid
    n_cols = 3
    n_rows = math.ceil(len(metric_keys) / n_cols)
    color_map = {lbl: _DS_COLORS[i % len(_DS_COLORS)] for i, lbl in enumerate(unique_labels)}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()
    x_labels = [label_names.get(int(lbl), str(lbl)) for lbl in unique_labels]

    for idx, k in enumerate(metric_keys):
        ax = axes_flat[idx]
        data_groups = [per_class_vals[k][lbl] for lbl in unique_labels]

        bp = ax.boxplot(data_groups, patch_artist=True, notch=False, widths=0.5,
                        medianprops=dict(color="black", linewidth=1.5))
        for patch, lbl in zip(bp["boxes"], unique_labels):
            patch.set_facecolor(color_map[lbl])
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(unique_labels) + 1))
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(k.replace("_", " "), fontsize=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

        pval_key = f"traj/anova_pvalue_{k}"
        if pval_key in anova_dict:
            pval = anova_dict[pval_key]
            stars = ("***" if pval < 0.001 else
                     "**" if pval < 0.01 else
                     "*" if pval < 0.05 else "n.s.")
            ax.set_xlabel(f"p = {pval:.2e}  {stars}", fontsize=8, color="grey")

    for idx in range(len(metric_keys), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Trajectory Geometry Boxplots — step {global_step}", fontsize=12)
    fig.tight_layout()
    return anova_dict, fig


def _plot_latent_pca_tokens(
    latent_tokens: np.ndarray,   # [N, T_tokens, D]
    labels: np.ndarray,          # [N]
    global_step: int,
    label_names: Optional[dict] = None,
) -> dict:
    """2D and 3D PCA scatter plots for token sequences [N, T_tokens, D].

    Always produces both with-trajectory-lines and without-lines variants.
    Keys: ``pca_2d_all_tokens``, ``pca_2d_all_tokens_lines``,
          ``pca_3d_all_tokens``, ``pca_3d_all_tokens_lines``.
    """
    from sklearn.decomposition import PCA

    N, T, D = latent_tokens.shape
    flat = latent_tokens.reshape(N * T, D)
    labels_flat = np.repeat(labels, T)

    unique_labels = np.unique(labels)
    if label_names is None:
        label_names = {l: str(l) for l in unique_labels}
    color_map = {l: _DS_COLORS[i % len(_DS_COLORS)] for i, l in enumerate(unique_labels)}

    figs: dict = {}

    pca2 = PCA(n_components=2)
    emb2 = pca2.fit_transform(flat)

    # ---- 2D: all labels overlaid ----
    fig2, ax = plt.subplots(figsize=(8, 6))
    for l in unique_labels:
        mask = labels_flat == l
        ax.scatter(emb2[mask, 0], emb2[mask, 1],
                   c=[color_map[l]], label=label_names.get(int(l), str(l)),
                   alpha=0.4, s=6, linewidths=0, zorder=2)
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"Latent PCA 2D (all tokens) — step {global_step}")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    figs["pca_2d_all_tokens"] = fig2

    # ---- 2D: per-label subfigures with trajectory lines ----
    n_cls = len(unique_labels)
    fig2l, axes2l = plt.subplots(1, n_cls, figsize=(7 * n_cls, 6), squeeze=False)
    arrow_stride = max(1, T // 5)
    for col, l in enumerate(unique_labels):
        ax = axes2l[0, col]
        class_indices = [i for i in range(N) if labels[i] == l]
        for i in class_indices:
            seg = emb2[i * T : (i + 1) * T]
            ax.plot(seg[:, 0], seg[:, 1],
                    color=color_map[l], alpha=0.25, linewidth=0.6, zorder=1)
            for t in range(0, T - 1, arrow_stride):
                ax.annotate(
                    "", xy=(seg[t + 1, 0], seg[t + 1, 1]),
                    xytext=(seg[t, 0], seg[t, 1]),
                    arrowprops=dict(
                        arrowstyle="->", color=color_map[l],
                        lw=0.8, alpha=0.5,
                    ),
                )
        mask = labels_flat == l
        ax.scatter(emb2[mask, 0], emb2[mask, 1],
                   c=[color_map[l]], alpha=0.4, s=6, linewidths=0, zorder=2)
        ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1] * 100:.1f}%)")
        ax.set_title(label_names.get(int(l), str(l)), fontsize=11)
        ax.grid(linestyle="--", linewidth=0.4, alpha=0.4)
    fig2l.suptitle(f"Latent PCA 2D trajectories — step {global_step}", fontsize=12)
    fig2l.tight_layout()
    figs["pca_2d_all_tokens_lines"] = fig2l

    if D >= 3:
        pca3 = PCA(n_components=3)
        emb3 = pca3.fit_transform(flat)

        # ---- 3D: all labels overlaid ----
        fig3 = plt.figure(figsize=(9, 7))
        ax3 = fig3.add_subplot(111, projection="3d")
        for l in unique_labels:
            mask = labels_flat == l
            ax3.scatter(emb3[mask, 0], emb3[mask, 1], emb3[mask, 2],
                        c=[color_map[l]], label=label_names.get(int(l), str(l)),
                        alpha=0.4, s=5, linewidths=0)
        ax3.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0] * 100:.1f}%)")
        ax3.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1] * 100:.1f}%)")
        ax3.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2] * 100:.1f}%)")
        ax3.set_title(f"Latent PCA 3D (all tokens) — step {global_step}")
        ax3.legend(fontsize=8)
        fig3.tight_layout()
        figs["pca_3d_all_tokens"] = fig3

        # ---- 3D: per-label subfigures with trajectory lines ----
        fig3l = plt.figure(figsize=(8 * n_cls, 7))
        arrow_stride3 = max(1, T // 5)
        for col, l in enumerate(unique_labels):
            ax3l = fig3l.add_subplot(1, n_cls, col + 1, projection="3d")
            class_indices = [i for i in range(N) if labels[i] == l]
            for i in class_indices:
                seg = emb3[i * T : (i + 1) * T]
                ax3l.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                          color=color_map[l], alpha=0.18, linewidth=0.6)
                for t in range(0, T - 1, arrow_stride3):
                    d = seg[t + 1] - seg[t]
                    ax3l.quiver(
                        seg[t, 0], seg[t, 1], seg[t, 2],
                        d[0], d[1], d[2],
                        color=color_map[l], alpha=0.4,
                        arrow_length_ratio=0.4, linewidth=0.5,
                    )
            mask = labels_flat == l
            ax3l.scatter(emb3[mask, 0], emb3[mask, 1], emb3[mask, 2],
                         c=[color_map[l]], alpha=0.4, s=5, linewidths=0)
            ax3l.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0] * 100:.1f}%)")
            ax3l.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1] * 100:.1f}%)")
            ax3l.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2] * 100:.1f}%)")
            ax3l.set_title(label_names.get(int(l), str(l)), fontsize=11)
        fig3l.suptitle(f"Latent PCA 3D trajectories — step {global_step}", fontsize=12)
        fig3l.tight_layout()
        figs["pca_3d_all_tokens_lines"] = fig3l

    return figs


@torch.no_grad()
def run_downstream_latent_analysis(
    model,
    val_loader,
    output_dir: str,
    global_step: int,
    n_classes: int = 3,
    label_names: Optional[dict] = None,
    max_samples: int = 512,
    device=None,
    support_bad_channel_mask: bool = False,
    pass_eeg_history: bool = False,
    knn_eval_cfg: Optional[dict] = None,
) -> dict:
    """Collect val-set latents and run PCA scatter + clustering quality metrics.

    Compatible with all three backbone types:
      - DLCNetMultichannel : last_hidden_state [B, D], hidden_states = None
      - EEGTransformer     : last_hidden_state [B, D], hidden_states [B, N_patches, D]
      - LaBraM             : last_hidden_state [B, D], hidden_states [B, N*A, D]

    Saves PNG plots to ``output_dir/latent_plots/`` and returns a flat dict of
    scalar metrics suitable for ``wandb.log``.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    per_class_max = max(1, max_samples // max(n_classes, 1))
    per_class: Dict[int, dict] = {}   # label → {"lhs": [tensors], "hs": [tensors], "trial": [int]}
    has_hidden_states: Optional[bool] = None   # determined from first batch

    for batch in val_loader:
        if (len(per_class) >= n_classes
                and all(len(v["lhs"]) >= per_class_max for v in per_class.values())):
            break

        eeg = batch["eeg"].float().to(device)
        labels_batch = batch["label"]
        trials_batch = batch.get("trial")
        bad_mask = batch.get("bad_channel_mask")
        if bad_mask is not None:
            bad_mask = bad_mask.to(device)
        hist_kw = {}
        if pass_eeg_history:
            eh = batch.get("eeg_history")
            if eh is not None:
                hist_kw["eeg_history"] = eh.float().to(device)

        if support_bad_channel_mask and bad_mask is not None:
            output = model(eeg, bad_channel_mask=bad_mask, **hist_kw)
        elif hist_kw:
            output = model(eeg, **hist_kw)
        else:
            output = model(eeg)
        lhs = output.last_hidden_state.float().cpu()   # [B, D]
        hs = output.hidden_states                       # [B, N_tokens, D] or None

        if has_hidden_states is None:
            has_hidden_states = hs is not None
        hs_cpu = hs.float().cpu() if (has_hidden_states and hs is not None) else None

        for i in range(lhs.shape[0]):
            lbl = int(labels_batch[i])
            if lbl not in per_class:
                per_class[lbl] = {"lhs": [], "hs": [], "trial": []}
            buf = per_class[lbl]
            if len(buf["lhs"]) < per_class_max:
                buf["lhs"].append(lhs[i])
                if trials_batch is not None:
                    buf["trial"].append(int(trials_batch[i]))
                if hs_cpu is not None:
                    buf["hs"].append(hs_cpu[i])   # [N_tokens, D]

    if not per_class:
        return {}

    # Concatenate balanced samples
    all_lhs: List[torch.Tensor] = []
    all_hs: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_trials: List[int] = []

    for lbl, buf in sorted(per_class.items()):
        all_lhs.append(torch.stack(buf["lhs"]))         # [n, D]
        if has_hidden_states and buf["hs"]:
            all_hs.append(torch.stack(buf["hs"]))       # [n, N_tokens, D]
        all_labels.extend([lbl] * len(buf["lhs"]))
        all_trials.extend(buf.get("trial", []))

    lhs_np = torch.cat(all_lhs, dim=0).numpy()          # [N, D]
    labels_arr = np.array(all_labels)                    # [N]
    trials_arr = np.array(all_trials) if all_trials else None

    plots_dir = os.path.join(output_dir, "latent_plots", f"step_{global_step:07d}")
    os.makedirs(plots_dir, exist_ok=True)

    wandb_log: dict = {}

    # ---- Pooled representation (last_hidden_state) ----
    cluster_metrics = _evaluate_latent_clustering(
        lhs_np,
        labels_arr,
        trial_arr=trials_arr,
        knn_eval_cfg=knn_eval_cfg,
    )
    for k, v in cluster_metrics.items():
        wandb_log[f"latent/cluster/{k}"] = v

    for name, fig in _plot_latent_pca(lhs_np, labels_arr, global_step, label_names).items():
        fig.savefig(os.path.join(plots_dir, f"{name}.png"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)

    # ---- Token-level representation (hidden_states, when available) ----
    if has_hidden_states and all_hs:
        hs_np = torch.cat(all_hs, dim=0).numpy()        # [N, N_tokens, D]
        N_s, T_tok, D_lat = hs_np.shape

        # ---- Trajectory geometry metrics + ANOVA + boxplots ----
        traj_metrics = _compute_trajectory_geometry(hs_np, labels_arr)
        for k, v in traj_metrics.items():
            wandb_log[f"latent/{k}"] = v
        traj_anova, traj_boxplot_fig = _plot_trajectory_geometry_boxplots(
            hs_np, labels_arr, global_step, label_names=label_names
        )
        for k, v in traj_anova.items():
            wandb_log[f"latent/{k}"] = v

        token_trials = np.repeat(trials_arr, T_tok) if trials_arr is not None else None
        cluster_all = _evaluate_latent_clustering(
            hs_np.reshape(N_s * T_tok, D_lat),
            np.repeat(labels_arr, T_tok),
            trial_arr=token_trials,
            knn_eval_cfg=knn_eval_cfg,
        )
        for k, v in cluster_all.items():
            wandb_log[f"latent/cluster_all_tokens/{k}"] = v

        for name, fig in _plot_latent_pca_tokens(
            hs_np, labels_arr, global_step, label_names
        ).items():
            fig.savefig(os.path.join(plots_dir, f"{name}.png"),
                        dpi=600, bbox_inches="tight")
            plt.close(fig)
        traj_boxplot_fig.savefig(
            os.path.join(plots_dir, "traj_boxplots.png"), dpi=600, bbox_inches="tight"
        )
        plt.close(traj_boxplot_fig)

    # Per-class sample counts
    for lbl, buf in per_class.items():
        wandb_log[f"latent/n_samples_class{lbl}"] = float(len(buf["lhs"]))

    return wandb_log


# ============================================================================
#  Main Training Loop
# ============================================================================
def main(config: dict):
    # ---- Unpack configuration sections ------------------------------------
    project_cfg = config["project"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    # ---- Seeds & environment -----------------------------------------------
    seed = project_cfg.get("seed", 42)
    set_seed(seed)

    # ---- WandB login (optional) -------------------------------------------
    # Prefer env WANDB_API_KEY (or WANDB_KEY) so the key is not stored in config
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")
    if not wandb_key:
        wandb_key = project_cfg.get("wandb_key")
    if wandb_key:
        wandb.login(key=wandb_key)

    # ---- Accelerator -------------------------------------------------------
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True, static_graph=False
    )
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs, timeout_kwargs],
    )

    # ---- Output directory ---------------------------------------------------
    group_name = project_cfg.get("group_name", "eeg_downstream")
    working_dir = project_cfg.get("working_dir", "./")
    resume_from_checkpoint = train_cfg.get("resume_from_checkpoint")

    if accelerator.is_main_process:
        if resume_from_checkpoint:
            output_path = resume_from_checkpoint
        else:
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            output_path = os.path.join(working_dir, "results", group_name, timestamp)
            if os.path.exists(output_path):
                output_path = output_path + "_new"
        os.makedirs(output_path, exist_ok=True)

        # Save a copy of the config
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        accelerator.init_trackers(
            project_cfg.get("wandb_project", "eeg-downstream"),
            config=config,
            init_kwargs={
                "wandb": {
                    "group": group_name,
                    "reinit": True,
                    "save_code": True,
                    "notes": os.environ.get("SLURM_JOB_ID", ""),
                }
            },
        )
    else:
        output_path = os.path.join(working_dir, "results", group_name)

    # ---- Datasets & DataLoaders ---------------------------------------------
    print("[train] Creating datasets ...")
    train_dataset, val_dataset = get_dataset(config)
    print(f"[train] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    batch_size = train_cfg.get("batch_size", 32)
    val_batch_size = train_cfg.get("val_batch_size", 32)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=safe_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=False,
        collate_fn=safe_collate_batch,
    )

    # ---- Model --------------------------------------------------------------
    print("[train] Building model ...")
    freeze_backbone = train_cfg.get("freeze_backbone", False)
    out = get_model(config, return_loaded_keys=freeze_backbone)
    if isinstance(out, tuple):
        model, loaded_keys = out
    else:
        model, loaded_keys = out, None
    model_summary(model)

    # ---- Freeze backbone (optional): freeze params loaded from checkpoint -----
    if freeze_backbone:
        trainable_parameters = train_cfg.get("trainable_parameters", [])

        def _param_kept_trainable(name: str) -> bool:
            if not trainable_parameters:
                return False
            return any(name == p or name.startswith(p) for p in trainable_parameters)

        pretrained_path = model_cfg.get("pretrained_path")
        if not loaded_keys and pretrained_path:
            if accelerator.is_main_process:
                print(
                    "[train] Warning: freeze_backbone=True but could not infer loaded "
                    "parameters from checkpoint (e.g. _load_pretrained path); not freezing."
                )
        else:
            for name, param in model.named_parameters():
                if name in loaded_keys and not _param_kept_trainable(name):
                    param.requires_grad_(False)
            frozen_names = [n for n, p in model.named_parameters() if not p.requires_grad]
            trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
            frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if accelerator.is_main_process:
                print("[train] Frozen parameters (loaded from checkpoint):")
                for n in sorted(frozen_names):
                    print(f"  {n}")
                print(f"[train] Frozen total: {len(frozen_names)} parameters, {frozen_count:,} elements")
                print("[train] Trainable parameters:")
                for n in sorted(trainable_names):
                    print(f"  {n}")
                print(f"[train] Trainable total: {len(trainable_names)} parameters, {trainable_count:,} elements")

    # ---- Un-freeze explicitly listed modules (override freeze_backbone) -----
    trainable_modules = train_cfg.get("trainable_modules", [])
    if trainable_modules:
        for module_prefix in trainable_modules:
            for name, param in model.named_parameters():
                if name.startswith(module_prefix):
                    param.requires_grad_(True)
        if accelerator.is_main_process:
            retrainable = [n for n, p in model.named_parameters() if p.requires_grad]
            print("[train] After trainable_modules override, trainable parameters:")
            for n in sorted(retrainable):
                print(f"  {n}")

    # ---- Loss ---------------------------------------------------------------
    loss_fn = get_loss_fn(config)

    # ---- Optimizer ----------------------------------------------------------
    learning_rate = train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.05)
    params = add_weight_decay(model, weight_decay=weight_decay)
    if accelerator.is_main_process:
        wandb.watch(model, log="all", log_freq=train_cfg.get("validation_steps", 100))

    optimizer_name = train_cfg.get("optimizer", "adamw").lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # ---- LR Scheduler -------------------------------------------------------
    gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)
    max_train_steps = train_cfg.get("max_train_steps", 10000)
    lr_warmup_steps = train_cfg.get("lr_warmup_steps", 500)
    scheduler_name = train_cfg.get("lr_scheduler", "cosine")

    lr_scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # ---- Prepare with Accelerator -------------------------------------------
    # Keep a reference to the raw val loader for latent analysis (main process only).
    raw_val_loader = val_loader
    model, loss_fn, optimizer, train_loader, val_loader, lr_scheduler = (
        accelerator.prepare(
            model, loss_fn, optimizer, train_loader, val_loader, lr_scheduler
        )
    )

    # ---- Training bookkeeping -----------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = (
        batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    validation_steps = train_cfg.get("validation_steps", 500)
    save_every_steps = train_cfg.get("save_every_steps", 1000)

    task_type = train_cfg.get("task_type", "regression")
    is_binary = train_cfg.get("is_binary", False)
    best_metric_name = train_cfg.get("best_metric", "val_loss")
    best_metric_higher = best_metric_name.replace("val_", "") in _HIGHER_IS_BETTER

    print("=" * 60)
    print("  Running downstream training")
    print(f"  Model:              {model_cfg['name']}")
    print(f"  Dataset:            {data_cfg['dataset']}")
    print(f"  Task type:          {task_type}")
    print(f"  Loss:               {train_cfg.get('loss', 'mse')}")
    if task_type == "classification":
        print(f"  Binary:             {is_binary}")
        print(f"  Metrics:            {train_cfg.get('metrics', [])}")
        print(f"  Best metric:        {best_metric_name}")
    print(f"  Train samples:      {len(train_dataset)}")
    print(f"  Val samples:        {len(val_dataset)}")
    print(f"  Epochs:             {num_train_epochs}")
    print(f"  Batch size / GPU:   {batch_size}")
    print(f"  Total batch size:   {total_batch_size}")
    print(f"  Grad accum steps:   {gradient_accumulation_steps}")
    print(f"  Total opt steps:    {max_train_steps}")
    print(f"  Learning rate:      {learning_rate}")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu_mem_info()

    global_step = 0
    first_epoch = 0
    best_metric_value = -float("inf") if best_metric_higher else float("inf")

    ckp_path = os.path.join(output_path, "ckp") if accelerator.is_main_process else ""
    best_path = os.path.join(output_path, "best") if accelerator.is_main_process else ""
    if accelerator.is_main_process:
        os.makedirs(ckp_path, exist_ok=True)
        os.makedirs(best_path, exist_ok=True)

    # ---- Resume from checkpoint ---------------------------------------------
    resume_step = 0
    if resume_from_checkpoint:
        dirs = [
            d
            for d in os.listdir(ckp_path)
            if d.startswith("checkpoint")
        ] if os.path.isdir(ckp_path) else []
        if dirs:
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            latest = dirs[-1]
            accelerator.print(f"[train] Resuming from checkpoint {latest}")
            accelerator.load_state(os.path.join(ckp_path, latest))
            global_step = int(latest.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
        else:
            accelerator.print("[train] No checkpoint found, starting from scratch")

    # ---- Progress bar -------------------------------------------------------
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    mst_sfreq = float(model_cfg.get("config", {}).get("sfreq", 200.0))
    is_mst_downstream = model_cfg.get("name", "").lower() == "mst_downstream"

    # ---- Training loop -------------------------------------------------------
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        epoch_loss_list = []

        for step, batch in enumerate(train_loader):
            # Skip steps for checkpoint resume
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(model):
                eeg = batch["eeg"].float()
                target = extract_target(batch, data_cfg, train_cfg)

                if is_mst_downstream:
                    bad = batch.get("bad_channel_mask")
                    hist_kw = _mst_eeg_history_kw(batch)
                    aug_cfg = train_cfg.get("mst_augmentation", {})
                    if aug_cfg.get("enabled", True):
                        rng = np.random.default_rng(
                            seed + global_step * 10007 + step + epoch * 1_000_003
                        )
                        probs_raw = aug_cfg.get("probs") or {}
                        probs = {
                            k: v
                            for k, v in probs_raw.items()
                            if v is not None
                        }
                        il_cfg = aug_cfg.get("intralabel_random_interpolation") or {}
                        il_on = bool(
                            il_cfg.get(
                                "enable_intralabel_random_interpolation", False
                            )
                        )
                        il_prob = float(il_cfg.get("intralabel_interp_prob", 0.0))
                        il_amin = float(
                            il_cfg.get("intralabel_interp_alpha_min", 0.2)
                        )
                        il_amax = float(
                            il_cfg.get("intralabel_interp_alpha_max", 0.8)
                        )
                        il_freq = bool(
                            il_cfg.get("intralabel_frequency_domain", False)
                        )
                        cg_log_std = float(
                            aug_cfg.get("channel_gain_log_std", 0.3)
                        )
                        tilt_max = float(
                            aug_cfg.get("spectral_tilt_max_slope", 0.3)
                        )
                        tilt_p = float(probs.get("spectral_tilt", 0.0))
                        pw_cfg = aug_cfg.get("post_wavelet") or {}
                        pw_il = pw_cfg.get("intralabel_interpolation") or {}
                        pw_tr = pw_cfg.get("time_roll") or {}
                        pw_il_en = bool(pw_il.get("enabled", False))
                        pw_il_p = float(pw_il.get("apply_prob", 0.5))
                        pw_il_amin = float(pw_il.get("alpha_min", 0.2))
                        pw_il_amax = float(pw_il.get("alpha_max", 0.8))
                        pw_tr_en = bool(pw_tr.get("enabled", False))
                        pw_tr_p = float(pw_tr.get("apply_prob", 0.5))
                        lab_mix = None
                        ds_mix = None
                        if task_type == "classification":
                            lab_mix = target.long().view(-1)
                            ds_mix = batch.get("dataset_id")
                            if ds_mix is None:
                                ds_mix = torch.zeros(
                                    eeg.shape[0],
                                    dtype=torch.long,
                                    device=eeg.device,
                                )
                            else:
                                ds_mix = ds_mix.long().view(-1)
                        eeg_aug, ch_mask = apply_augmentation_view_downstream(
                            eeg,
                            sfreq=mst_sfreq,
                            rng=rng,
                            probs=probs or None,
                            labels=lab_mix,
                            dataset_ids=ds_mix,
                            intralabel_enabled=il_on,
                            intralabel_apply_prob=il_prob,
                            intralabel_alpha_min=il_amin,
                            intralabel_alpha_max=il_amax,
                            intralabel_frequency_domain=il_freq,
                            channel_gain_log_std=cg_log_std,
                        )
                        output = model(
                            eeg_aug,
                            bad_channel_mask=bad,
                            channel_mask=ch_mask,
                            augmentation_rng=rng,
                            spectral_tilt_prob=tilt_p,
                            spectral_tilt_max_slope=tilt_max,
                            labels=lab_mix,
                            dataset_ids=ds_mix,
                            post_wavelet_intralabel_enabled=pw_il_en,
                            post_wavelet_intralabel_apply_prob=pw_il_p,
                            post_wavelet_intralabel_alpha_min=pw_il_amin,
                            post_wavelet_intralabel_alpha_max=pw_il_amax,
                            post_wavelet_time_roll_enabled=pw_tr_en,
                            post_wavelet_time_roll_apply_prob=pw_tr_p,
                            **hist_kw,
                        )
                    else:
                        output = model(eeg, bad_channel_mask=bad, **hist_kw)
                else:
                    output = model(eeg)
                preds = output.predictions
                # Cast logits to float32 for loss to avoid bf16 overflow in log_softmax (NaN loss)
                if preds.dtype != torch.float32:
                    preds = preds.float()

                if task_type != "classification" and preds.shape != target.shape:
                    target = target[..., :preds.shape[-1]]
                loss = loss_fn(preds, target)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                epoch_loss_list.append(loss.item())

                logs = {
                    "train_loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                if getattr(output, "proto_loss", None) is not None:
                    logs["train_proto_loss"] = output.proto_loss.item()

                if task_type == "classification":
                    with torch.no_grad():
                        if is_binary:
                            pred_labels = (torch.sigmoid(preds) > 0.5).long()
                            true_labels = target.long()
                        else:
                            pred_labels = preds.argmax(dim=-1)
                            true_labels = target
                        train_acc = (pred_labels.squeeze() == true_labels.squeeze()).float().mean().item()
                    logs["train_acc"] = train_acc

                progress_bar.set_postfix(**logs)

                # if accelerator.is_main_process:
                #     accelerator.log(logs, step=global_step)

                # ---- Validation -------------------------------------------------
                if global_step % validation_steps == 0:
                    val_metrics = validate(
                        model, val_loader, loss_fn, data_cfg, train_cfg,
                        accelerator, global_step,
                        current_lr=lr_scheduler.get_last_lr()[0],
                        model_cfg=model_cfg,
                    )
                    metric_strs = [f"val_loss={val_metrics['val_loss']:.6f}"]
                    for k, v in val_metrics.items():
                        if k != "val_loss":
                            metric_strs.append(f"{k}={v:.4f}")
                    accelerator.print(
                        f"\n[step {global_step}] " + "  ".join(metric_strs)
                    )
                    if val_metrics.get("val_proto_loss") is not None:
                        accelerator.print(
                            f"  proto_loss={val_metrics['val_proto_loss']:.6f}"
                        )

                    # Save best model
                    current_metric = val_metrics.get(best_metric_name, val_metrics["val_loss"])
                    improved = (
                        (current_metric > best_metric_value) if best_metric_higher
                        else (current_metric < best_metric_value)
                    )
                    if improved and accelerator.is_main_process:
                        best_metric_value = current_metric
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(best_path)
                        accelerator.print(
                            f"[train] New best model saved ({best_metric_name}={best_metric_value:.6f})"
                        )

                    # ---- Latent analysis ----------------------------------------
                    if accelerator.is_main_process and task_type == "classification":
                        _la_cfg = train_cfg.get("latent_analysis", {})
                        if _la_cfg.get("enabled", True):
                            _raw_names = _la_cfg.get("label_names")
                            _lnames = (
                                {int(k): v for k, v in _raw_names.items()}
                                if _raw_names else None
                            )
                            _n_cls = model_cfg["config"].get("downstream_output_dim", 3)
                            _unwrapped = accelerator.unwrap_model(model)
                            latent_log = run_downstream_latent_analysis(
                                _unwrapped,
                                raw_val_loader,
                                output_path,
                                global_step,
                                n_classes=_n_cls,
                                label_names=_lnames,
                                max_samples=_la_cfg.get("max_samples", 512),
                                device=accelerator.device,
                                support_bad_channel_mask=(
                                    model_cfg.get("name") == "mst_downstream"
                                ),
                                pass_eeg_history=(
                                    model_cfg.get("name") == "mst_downstream"
                                ),
                                knn_eval_cfg=train_cfg.get("knn_eval", {}),
                            )
                            if latent_log:
                                wandb.log(latent_log, step=global_step)
                                accelerator.print(
                                    "[train] Latent: "
                                    + "  ".join(
                                        f"{k.split('/')[-1]}={v:.4f}"
                                        for k, v in latent_log.items()
                                        if isinstance(v, float) and "cluster" in k
                                    )
                                )

                    model.train()

                # ---- Periodic checkpoint ----------------------------------------
                if (
                    global_step % save_every_steps == 0
                    and accelerator.is_main_process
                ):
                    save_dir = os.path.join(ckp_path, f"checkpoint-{global_step}")
                    accelerator.save_state(save_dir)
                    # Keep only the last 1 checkpoints
                    _cleanup_checkpoints(ckp_path, keep=1)

            if global_step >= max_train_steps:
                break

        # End-of-epoch logging
        if epoch_loss_list and accelerator.is_main_process:
            accelerator.log(
                {
                    "epoch_train_loss": np.mean(epoch_loss_list),
                    "epoch": epoch,
                },
                step=global_step,
            )

        if global_step >= max_train_steps:
            break

    # ---- Cleanup & final save -----------------------------------------------
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_path = os.path.join(output_path, "final")
        os.makedirs(final_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_path)
        accelerator.print(f"[train] Final model saved to {final_path}")
        accelerator.print(f"[train] Best model saved to {best_path}")
        accelerator.print(f"[train] Best {best_metric_name} = {best_metric_value:.6f}")

    accelerator.end_training()


# ============================================================================
#  Helpers
# ============================================================================
def _cleanup_checkpoints(ckp_dir: str, keep: int = 2):
    """Keep only the *keep* most recent checkpoints."""
    if not os.path.isdir(ckp_dir):
        return
    dirs = [d for d in os.listdir(ckp_dir) if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    import shutil
    for d in dirs[:-keep]:
        shutil.rmtree(os.path.join(ckp_dir, d), ignore_errors=True)


# ============================================================================
#  CLI entry point
# ============================================================================
def get_args_parser():
    parser = argparse.ArgumentParser("EEG Downstream Task Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the JSON configuration file",
    )
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # Store config path for reference
    config["_config_path"] = args.config

    main(config)
