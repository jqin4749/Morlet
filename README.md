# Morlet Spectral Transformer (MST)

Official implementation of **Dive into Waves: Morlet Spectral Transformer for Cross-Subject Emotion Decoding from EEG**.

This repository trains **MST** for cross-subject EEG emotion recognition on the **SEED-family** benchmarks (SEED, SEED-IV, SEED-V, SEED-VII). The model does not rely on large-scale EEG pretraining; instead it uses neuroscience-informed time–frequency tokenization and a spatiotemporal Transformer backbone.

## Overview

Emotion-related EEG is weak, noisy, and highly variable across subjects. MST targets three representation bottlenecks:

1. **Morlet wavelet tokenization** — A fixed bank of complex Morlet wavelets (20 log-spaced bands from 2–45 Hz) converts each 2 s window into a dense time–frequency grid. This generalizes classical differential-entropy (DE) features to a multi-resolution representation suitable for Transformers.
2. **Long-context baseline removal** — Neighboring segments supply a local spectral baseline; the model sees both raw log-power and residual (raw − baseline), suppressing subject-specific drift and temporal redundancy.
3. **Frequency-specific spatial projection** — A separate learned channel mixer per frequency band captures band-dependent scalp topographies before self-attention.

The backbone stacks pre-LayerNorm Transformer blocks with **2D RoPE** over the frequency × time token grid. Classification uses a **dual-pooling head** (concatenated CLS token and mean-pooled patch tokens).

```
Raw EEG (B × T × C)
    → Morlet convolution + temporal pooling → spectrogram tokens
    → optional baseline removal (dual branch)
    → per-frequency spatial projection → patch embeddings
    → Transformer (2D RoPE) → CLS + mean pool → classifier
```

| Component | Default (paper / example config) |
|-----------|----------------------------------|
| Channels | 62 (SEED-family montage) |
| Window | 400 samples @ 200 Hz (2 s) |
| Morlet bands | 20 (2–45 Hz, log-spaced) |
| Time bins | 16 |
| Transformer | 12 layers, 256 dim, 8 heads |
| Baseline context | 5 prior segments (~20 s) |

## Supported datasets

| Dataset | Classes | Subjects | Notes |
|---------|---------|----------|-------|
| SEED | 3 | 15 | `seed` |
| SEED-IV | 4 | 15 | `seediv` |
| SEED-V | 5 | 16 | `seedv` |
| SEED-VII | 7 | 20 | `seedvii` |

Datasets must be obtained from the [BCMI lab (SJTU)](https://bcmi.sjtu.edu.cn/home/seed/) under their license agreement (academic use only; no redistribution).

**Preprocessing** (implemented in `morlet/datasets/*_preprocess.py`, matching the paper appendix):

- Band-pass 0.1–75 Hz, 50 Hz notch, downsample to 200 Hz  
- 62 channels in a consistent order  
- Per-subject per-channel z-score normalization before windowing  
- Non-overlapping 2 s segments with clip-level emotion labels  

**Evaluation:** The paper reports strict **leave-one-subject-out (LOSO)** cross-validation. For each fold, hold out one subject for validation and train on the rest by setting `val_subjects` when generating splits (see [Data preparation](#data-preparation)). Repeat for every subject and aggregate metrics across folds.

## Installation

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.1.

**Option A — Conda**

```bash
conda env create -f environment.yml
conda activate morlet
# Install a CUDA-enabled PyTorch build if needed: https://pytorch.org
```

**Option B — venv + pip**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Data preparation

### 1. Preprocess raw recordings

Each dataset module exposes `preprocess_*_to_npy`. Example for SEED:

```python
from morlet.datasets.seed_preprocess import preprocess_seed_to_npy

preprocess_seed_to_npy(
    dataset_path="/path/to/SEED",  # contains Chinese/01-EEG-raw/
    output_root=None,              # default: .../Chinese/preprocessed
)
```

Analogous entry points:

- `preprocess_seediv_to_npy` — SEED-IV  
- `preprocess_seedv_to_npy` — SEED-V  
- `preprocess_seedvii_to_npy` — SEED-VII  

Each trial is written as a directory:

```
preprocessed/
  sub1_sess1_clip01/
    eeg.npy          # (n_samples, 62) float32, µV
    timestamps.npy
    meta.json        # subject_id, session_id, label, ...
```

### 2. Generate train/validation splits

```python
from morlet.datasets.seed_preprocess import generate_seed_splits

# Single fold: hold out subject "15" for validation (LOSO)
generate_seed_splits(
    preprocessed_path="/path/to/seed/Chinese/preprocessed",
    mode="subject_independent",
    val_subjects=["15"],
    output_path="/path/to/seed/Chinese/preprocessed/splits_loso_sub15.json",
)
```

Split JSON files list either `train_subjects` / `val_subjects` or `train_trial_ids` / `val_trial_ids`. Point `splits_path` in your training config at the file for the current fold.

Optional **video-per-class** splits (`generate_*_video_per_class_splits`) select one clip per emotion class from held-out subjects for a harder generalization setting.

## Training

Edit `configs/mst_seed_subject_independent.json` (or copy it per dataset/fold) and set:

- `data.dataset_path` — preprocessed root  
- `data.splits_path` — split JSON for this fold  
- `data.normalize` — `"subject"` for per-subject z-score (paper default)  
- `project.wandb_key` or disable W&B in the script if not used  

```bash
cd Morlet
pip install -e .
export WANDB_API_KEY=...   # optional

python scripts/downstream.py --config configs/mst_seed_subject_independent.json
```

**Multi-GPU:**

```bash
accelerate launch scripts/downstream.py --config configs/mst_seed_subject_independent.json
```

Training runs until `max_train_steps` (default 50,000) or early stopping by validation; the best checkpoint is selected by `training.best_metric` (default: `val_balanced_accuracy`). Checkpoints and logs are written under the experiment group name in `project.group_name`.

### Key configuration fields

| Section | Field | Role |
|---------|-------|------|
| `model.name` | `mst_downstream` | Required model identifier |
| `model.config.freq_baseline_n_segments` | e.g. `5` | Prior segments for baseline removal; dataset supplies `eeg_history` when > 0 |
| `model.config.n_freqs`, `n_time_frames` | `20`, `16` | Morlet grid size |
| `data.interval_length` | `400` | Samples per window @ 200 Hz |
| `training.mst_augmentation` | see config | Phase noise, band noise, channel dropout, wavelet time roll, etc. |

When `freq_baseline_n_segments` in the model config matches the dataloader setting, batches include `eeg_history` for long-context baseline removal at the spectrogram level.

## Repository layout

```
Morlet/
├── morlet/
│   ├── models/
│   │   ├── mst.py           # Morlet front-end, spatial projection, Transformer, head
│   │   └── types.py         # ModelOutput dataclass
│   ├── datasets/
│   │   ├── seed.py          # PyTorch datasets + windowing
│   │   ├── seed_preprocess.py
│   │   └── ...              # SEED-IV / SEED-V / SEED-VII counterparts
│   └── utils/
│       ├── augmentation.py  # Training-time augmentations
│       └── misc.py          # Optimizer helpers, logging utilities
├── scripts/
│   └── downstream.py        # Classification training loop (Accelerate + W&B)
├── configs/
│   └── mst_seed_subject_independent.json
├── environment.yml
├── requirements.txt
└── pyproject.toml
```

## Citation

If you use this code, please cite:

```bibtex
@article{qing2026morlet,
  title={Dive into Waves: Morlet Spectral Transformer for Cross-Subject Emotion Decoding from EEG},
  author={Qing, Jiaxin and Li, Lexin},
  year={2026}
}
```

Also cite the SEED-family dataset papers as required by the [SJTU license](https://bcmi.sjtu.edu.cn/home/seed/).

## License

Released under the Apache License 2.0 (see `pyproject.toml`).
