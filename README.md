# Morlet (MST) Code Release

This repository contains a minimal codebase for training the **Morlet Spectral Transformer (MST)** downstream model from the paper *Dive into Waves: Morlet Spectral Transformer for Cross-Subject Emotion Decoding from EEG*: Morlet wavelet front-end, long-context baseline removal, frequency-specific spatial projection, a 2D RoPE Transformer, and **SEED / SEED-IV / SEED-V / SEED-VII** data loaders plus preprocessing entry points.


## Environment

**Option A: `conda`**

```bash
conda env create -f environment.yml
conda activate morlet
# For GPU: install a PyTorch build that matches your CUDA version from https://pytorch.org
```

**Option B: `venv` + pip**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Data

Follow the instructions in each `seed*_preprocess.py` to convert raw SEED-family data into trial directories with `eeg.npy`, `meta.json`, etc., and to produce `splits*.json`. Point `dataset_path` and `splits_path` in your JSON config at the root of that preprocessed layout.

## Training

Single-machine example (set `WANDB_API_KEY` if you use Weights & Biases, or disable logging in config):

```bash
cd Morlet
pip install -e .
python scripts/downstream.py --config configs/mst_seed_subject_independent.json
```

For multiple GPUs, use `accelerate launch` as in the upstream Accelerate workflow.

## Configuration

- `model.name` must be **`mst_downstream`**.
- Long-context baseline: when `model.config.freq_baseline_n_segments` matches the data config, the dataset supplies `eeg_history`. Training-time augmentation is controlled via **`mst_augmentation`**.
- Example configs:
  - `configs/mst_seed_subject_independent.json` — SEED, subject-independent split
  - `configs/mst_seedv_subject_dependent.json` — SEED-V
  - `configs/mst_seedvii_subject_dependent.json` — SEED-VII

## Repository layout

```
Morlet/
  morlet/
    models/mst.py        # MST downstream model and Morlet front-end
    datasets/              # SEED-family datasets and preprocessing helpers
    utils/                 # Augmentations and training utilities
  scripts/downstream.py    # Downstream training (trimmed from upstream eegmodel)
  configs/                 # Example JSON configs
```