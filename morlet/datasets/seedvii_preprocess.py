from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np

from .seed_preprocess import (
    SEED_STANDARD_CHANNELS,
    _write_trial,
    _generate_video_per_class_splits_impl,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDVII_LABEL_NAMES: Dict[int, str] = {
    0: "disgust",
    1: "fear",
    2: "sad",
    3: "neutral",
    4: "happy",
    5: "anger",
    6: "surprise",
}

NUM_CLASSES = 7

# Per-session clip emotion labels (derived from emotion_label_and_stimuli_order.xlsx).
# Each session has 20 clips; clip indices 1-20 = session 1, 21-40 = session 2, etc.
SEEDVII_SESSION_LABELS: Dict[int, List[int]] = {
    1: [4, 3, 0, 2, 5, 5, 2, 0, 3, 4, 4, 3, 0, 2, 5, 5, 2, 0, 3, 4],
    2: [5, 2, 1, 3, 6, 6, 3, 1, 2, 5, 5, 2, 1, 3, 6, 6, 3, 1, 2, 5],
    3: [4, 6, 0, 1, 5, 5, 1, 0, 6, 4, 4, 6, 0, 1, 5, 5, 1, 0, 6, 4],
    4: [0, 2, 1, 6, 4, 4, 6, 1, 2, 0, 0, 2, 1, 6, 4, 4, 6, 1, 2, 0],
}

SAMPLING_FREQUENCY = 200.0
NUM_SESSIONS = 4
CLIPS_PER_SESSION = 20
TOTAL_CLIPS = NUM_SESSIONS * CLIPS_PER_SESSION  # 80


# ===================================================================
# Full preprocessing: SEED-VII preprocessed .mat -> per-clip npy
# ===================================================================

def preprocess_seedvii_to_npy(
    dataset_path: str,
    output_root: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Convert SEED-VII preprocessed ``.mat`` files into per-clip npy trial dirs.

    The EEG data in SEED-VII is already preprocessed (bandpass filtered,
    62 channels, 200 Hz), so no filtering or resampling is performed.
    This function only reformats from MATLAB to the standard npy trial
    directory layout used by the rest of the pipeline.

    Parameters
    ----------
    dataset_path : str
        Path to the SEED-VII root (containing ``EEG_preprocessed/``).
    output_root : str, optional
        Where to write the preprocessed output.  Defaults to
        ``<dataset_path>/preprocessed``.
    overwrite : bool
        Re-process trials that already exist on disk.

    Returns
    -------
    str
        The *output_root* path.
    """
    import scipy.io

    eeg_dir = os.path.join(dataset_path, "EEG_preprocessed")
    if not os.path.isdir(eeg_dir):
        raise FileNotFoundError(
            f"Expected EEG_preprocessed directory at {eeg_dir}"
        )

    if output_root is None:
        output_root = os.path.join(dataset_path, "preprocessed")
    os.makedirs(output_root, exist_ok=True)

    mat_files = sorted(
        (f for f in os.listdir(eeg_dir) if f.endswith(".mat")),
        key=lambda f: int(f[:-4]),
    )
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {eeg_dir}")

    n_ok, n_skip, n_err = 0, 0, 0

    for mat_name in mat_files:
        subject_id = mat_name[:-4]  # e.g. "1", "20"
        mat_path = os.path.join(eeg_dir, mat_name)

        print(
            f"[seedvii] Processing {mat_name} (subject={subject_id}) ..."
        )

        try:
            mat_data = scipy.io.loadmat(mat_path)
        except Exception as exc:
            warnings.warn(f"Skipping {mat_path}: {exc}")
            n_err += 1
            continue

        for clip_idx in range(1, TOTAL_CLIPS + 1):
            clip_key = str(clip_idx)
            if clip_key not in mat_data:
                warnings.warn(
                    f"Missing clip key '{clip_key}' in {mat_name}"
                )
                n_err += 1
                continue

            session_id = (clip_idx - 1) // CLIPS_PER_SESSION + 1
            within_session_pos = (clip_idx - 1) % CLIPS_PER_SESSION
            clip_num = within_session_pos + 1

            trial_id = f"sub{subject_id}_sess{session_id}_clip{clip_num:02d}"
            out_dir = os.path.join(output_root, trial_id)

            if (
                not overwrite
                and os.path.isfile(os.path.join(out_dir, "meta.json"))
            ):
                n_skip += 1
                continue

            # EEG: (62, N) float64 -> (N, 62) float32
            eeg_raw = mat_data[clip_key]
            if eeg_raw.ndim != 2 or eeg_raw.shape[0] != len(SEED_STANDARD_CHANNELS):
                warnings.warn(
                    f"Unexpected shape {eeg_raw.shape} for clip {clip_key} "
                    f"in {mat_name}, expected (62, N)"
                )
                n_err += 1
                continue

            eeg = eeg_raw.T.astype(np.float32)  # (N, 62)
            n_samples = eeg.shape[0]
            timestamps = (np.arange(n_samples) / SAMPLING_FREQUENCY).astype(np.float32)

            label = SEEDVII_SESSION_LABELS[session_id][within_session_pos]

            meta = {
                "trial_id": trial_id,
                "subject_id": subject_id,
                "session_id": session_id,
                "clip_id": clip_num,
                "global_clip_index": clip_idx,
                "label": label,
                "label_name": SEEDVII_LABEL_NAMES.get(label, "unknown"),
                "num_samples": int(n_samples),
                "num_eeg_channels": int(eeg.shape[1]),
                "sampling_frequency": SAMPLING_FREQUENCY,
                "channel_labels": list(SEED_STANDARD_CHANNELS),
            }

            _write_trial(out_dir, eeg, timestamps, meta, overwrite=overwrite)
            n_ok += 1

        if (n_ok + n_err) % 100 == 0 and n_ok + n_err > 0:
            print(f"  progress: {n_ok} ok, {n_err} err, {n_skip} skip ...")

    print(
        f"[seedvii] Done: {n_ok} clips written, {n_err} errors, "
        f"{n_skip} skipped -> {output_root}"
    )
    return output_root


# ===================================================================
# Split generation
# ===================================================================

def generate_seedvii_splits(
    preprocessed_path: str,
    mode: str = "subject_independent",
    val_ratio: float = 0.2,
    seed: int = 42,
    val_subjects: Optional[Sequence[str]] = None,
    train_sessions: Optional[Sequence[int]] = None,
    val_sessions: Optional[Sequence[int]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Create a ``splits.json`` for the SEED-VII preprocessed directory.

    Parameters
    ----------
    preprocessed_path : str
        Path containing the trial directories (output of
        :func:`preprocess_seedvii_to_npy`).
    mode : str
        ``"subject_independent"`` -- split by subject, or
        ``"subject_dependent"``  -- split by session.
    val_ratio : float
        Fraction of subjects held out for validation (subject-independent
        mode only, ignored when *val_subjects* is given).
    seed : int
        Random seed for deterministic shuffling.
    val_subjects : sequence of str, optional
        Explicit list of validation subject IDs (overrides *val_ratio*).
    train_sessions / val_sessions : sequence of int, optional
        Explicit session lists (subject-dependent mode).
    output_path : str, optional
        Where to write the JSON.  Defaults to
        ``<preprocessed_path>/splits_<mode>.json``.

    Returns
    -------
    str
        Path to the written JSON.
    """
    subjects: set = set()
    sessions: set = set()
    for name in sorted(os.listdir(preprocessed_path)):
        meta_path = os.path.join(preprocessed_path, name, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        subjects.add(str(meta["subject_id"]))
        sessions.add(int(meta["session_id"]))

    subjects_sorted = sorted(subjects, key=lambda s: int(s))
    sessions_sorted = sorted(sessions)

    if output_path is None:
        output_path = os.path.join(
            preprocessed_path, f"splits_{mode}.json"
        )

    if mode == "subject_independent":
        if val_subjects is not None:
            val_set = {str(s) for s in val_subjects}
            train_set = set(subjects_sorted) - val_set
        else:
            rng = np.random.RandomState(seed)
            shuffled = list(subjects_sorted)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_ratio))
            val_set = set(shuffled[:n_val])
            train_set = set(shuffled[n_val:])

        splits = {
            "mode": "subject_independent",
            "seed": seed,
            "val_ratio": val_ratio,
            "n_subjects": len(subjects_sorted),
            "train_subjects": sorted(train_set, key=lambda s: int(s)),
            "val_subjects": sorted(val_set, key=lambda s: int(s)),
            "train_sessions": None,
            "val_sessions": None,
        }

    elif mode == "subject_dependent":
        t_sess = list(train_sessions) if train_sessions else [1, 2, 3]
        v_sess = list(val_sessions) if val_sessions else [4]
        splits = {
            "mode": "subject_dependent",
            "seed": seed,
            "n_subjects": len(subjects_sorted),
            "all_subjects": subjects_sorted,
            "train_subjects": None,
            "val_subjects": None,
            "train_sessions": t_sess,
            "val_sessions": v_sess,
        }
    else:
        raise ValueError(
            f"Unknown split mode '{mode}'. "
            "Use 'subject_independent' or 'subject_dependent'."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    print(f"[seedvii] Splits written: {output_path}")
    return output_path


def generate_seedvii_video_per_class_splits(
    preprocessed_path: str,
    independent_splits_path: str,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> str:
    """Generate a ``splits_video_per_class.json`` for the SEED-VII dataset.

    Uses the validation subjects from *independent_splits_path* and
    selects one clip per class for training; all remaining clips from
    those subjects form the validation set.

    Parameters
    ----------
    preprocessed_path : str
        Path to ``/data/seeds/seed-vii/preprocessed``.
    independent_splits_path : str
        Path to the existing ``splits_subject_independent.json``.
    seed : int
        Random seed (default 42).
    output_path : str, optional
        Output path.  Defaults to
        ``<preprocessed_path>/splits_video_per_class.json``.

    Returns
    -------
    str
        Path to the written JSON.
    """
    path = _generate_video_per_class_splits_impl(
        preprocessed_path, independent_splits_path, seed, output_path
    )
    print(f"[seedvii] Video-per-class splits written: {path}")
    return path
