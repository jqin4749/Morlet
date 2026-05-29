from __future__ import annotations

import json
import os
import re
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_STANDARD_CHANNELS: List[str] = [
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
]
_STANDARD_SET = frozenset(SEED_STANDARD_CHANNELS)

# 15 movie-clip emotion labels (same for every session):
#   1 = positive / happy,  0 = neutral,  -1 = negative / sad
SEED_CLIP_LABELS: List[int] = [
    1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1,
]

SEED_LABEL_NAMES: Dict[int, str] = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}

NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Time boundary parsing
# ---------------------------------------------------------------------------

def _parse_time_boundaries(time_txt_path: str) -> List[Tuple[int, int]]:
    """Parse ``time.txt`` → list of (start_sample, end_sample) at 1000 Hz."""
    with open(time_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    starts_m = re.search(r"start_point_list\s*=\s*\[([^\]]+)\]", text)
    ends_m = re.search(r"end_point_list\s*=\s*\[([^\]]+)\]", text)
    if not starts_m or not ends_m:
        raise ValueError(f"Cannot parse time boundaries from {time_txt_path}")

    starts = [int(x.strip()) for x in starts_m.group(1).split(",") if x.strip()]
    ends = [int(x.strip()) for x in ends_m.group(1).split(",") if x.strip()]
    if len(starts) != len(ends):
        raise ValueError("Mismatched start / end point counts in time.txt")
    return list(zip(starts, ends))


# ---------------------------------------------------------------------------
# Low-level Neuroscan CNT reader
# ---------------------------------------------------------------------------

def _read_cnt(cnt_path: str) -> Tuple[np.ndarray, List[str], float]:
    """Parse a Neuroscan ``.cnt`` (v4.x) and return raw data in micro-volts.

    MNE's ``read_raw_cnt`` overflows on the SEED ~1 GB files, so we read
    the binary directly.

    Returns ``(data_uv, ch_names, sfreq)`` where *data_uv* has shape
    ``(n_channels, n_samples)`` float32 and values are in micro-volts.
    """
    import struct

    with open(cnt_path, "rb") as f:
        header = f.read(900)

        nchannels = struct.unpack_from("<H", header, 370)[0]
        # Sample rate stored as uint16 at offset 376
        sfreq = float(struct.unpack_from("<H", header, 376)[0])
        event_pos = struct.unpack_from("<i", header, 886)[0]

        f.seek(0, 2)
        file_size = f.tell()

        # Channel headers (75 bytes each)
        f.seek(900)
        ch_names: List[str] = []
        sensitivities = np.empty(nchannels, dtype=np.float64)
        cal_gains = np.empty(nchannels, dtype=np.float64)
        for i in range(nchannels):
            ch_h = f.read(75)
            name = ch_h[:10].split(b"\x00")[0].decode("ascii", errors="ignore")
            ch_names.append(name.upper())
            sensitivities[i] = struct.unpack_from("<f", ch_h, 59)[0]
            cal_gains[i] = struct.unpack_from("<f", ch_h, 71)[0]

        data_offset = 900 + nchannels * 75
        if 0 < event_pos < file_size:
            data_bytes = event_pos - data_offset
        else:
            data_bytes = file_size - data_offset

        n_samples = data_bytes // (nchannels * 4)

        f.seek(data_offset)
        raw = np.frombuffer(
            f.read(n_samples * nchannels * 4), dtype="<i4",
        ).reshape(n_samples, nchannels)

    # Scale: uV = raw * sensitivity / (calibration * 204.8)
    cal_gains = np.where(cal_gains == 0, 1.0, cal_gains)
    scale = sensitivities / (cal_gains * 204.8)
    data_uv = raw.astype(np.float32) * scale.astype(np.float32)
    data_uv = data_uv.T  # (n_channels, n_samples) for MNE convention

    return data_uv, ch_names, sfreq


# ---------------------------------------------------------------------------
# CNT preprocessing
# ---------------------------------------------------------------------------

def _preprocess_cnt(
    cnt_path: str,
    target_sfreq: float = 200.0,
    band_low: float = 0.1,
    band_high: float = 75.0,
    notch_freq: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Read a Neuroscan ``.cnt``, standardise channels, filter, resample.

    Returns ``(eeg, timestamps, info)`` where *eeg* is
    ``(n_samples, n_channels)`` float32 in micro-volts.
    """
    import mne
    mne.set_log_level("ERROR")

    data_uv, ch_names, original_sfreq = _read_cnt(cnt_path)

    # Build MNE Raw from the parsed data (already in uV → convert to V)
    mne_info = mne.create_info(
        ch_names=ch_names,
        sfreq=original_sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data_uv * 1e-6, mne_info)  # MNE expects volts

    # Keep only standard EEG channels
    available = set(raw.ch_names)
    missing_channels = [ch for ch in SEED_STANDARD_CHANNELS if ch not in available]
    if missing_channels:
        raise ValueError(
            f"Missing standard EEG channels: {missing_channels[:8]}"
            + (" ..." if len(missing_channels) > 8 else "")
        )
    keep_channels = list(SEED_STANDARD_CHANNELS)
    drop_channels = [ch for ch in raw.ch_names if ch not in _STANDARD_SET]

    if drop_channels:
        raw.drop_channels(drop_channels)
    if not keep_channels:
        raise ValueError(f"No standard EEG channels found in {cnt_path}")

    raw.reorder_channels(keep_channels)

    raw.filter(l_freq=band_low, h_freq=band_high)
    raw.notch_filter(notch_freq)
    raw.resample(target_sfreq)

    try:
        eeg = raw.get_data(units="uV").T.astype(np.float32)
    except TypeError:
        eeg = (raw.get_data() * 1e6).T.astype(np.float32)

    timestamps = raw.times.astype(np.float32)

    info = {
        "original_sfreq": original_sfreq,
        "sampling_frequency": float(target_sfreq),
        "num_samples": int(eeg.shape[0]),
        "num_eeg_channels": int(eeg.shape[1]),
        "channel_labels": list(keep_channels),
        "source_cnt": os.path.abspath(cnt_path),
        "bandpass_hz": [band_low, band_high],
        "notch_hz": notch_freq,
    }
    raw.close()
    return eeg, timestamps, info


# ---------------------------------------------------------------------------
# Write helper (mirrors tuh_preprocess._write_trial)
# ---------------------------------------------------------------------------

def _write_trial(
    out_dir: str,
    eeg: np.ndarray,
    timestamps: np.ndarray,
    meta: dict,
    overwrite: bool = False,
) -> bool:
    """Write ``eeg.npy``, ``timestamps.npy``, ``mean.npy``, ``std.npy``,
    ``meta.json`` for a single trial."""
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        k: os.path.join(out_dir, f"{k}.npy")
        for k in ("eeg", "timestamps", "mean", "std")
    }
    meta_path = os.path.join(out_dir, "meta.json")

    if (
        not overwrite
        and all(os.path.isfile(p) for p in paths.values())
        and os.path.isfile(meta_path)
    ):
        return False

    mean = eeg.mean(axis=0).astype(np.float32)
    std = eeg.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    np.save(paths["eeg"], eeg)
    np.save(paths["timestamps"], timestamps)
    np.save(paths["mean"], mean)
    np.save(paths["std"], std)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return True


# ===================================================================
# Full preprocessing: SEED raw CNT → per-clip npy
# ===================================================================

def preprocess_seed_to_npy(
    dataset_path: str,
    output_root: Optional[str] = None,
    target_sfreq: float = 200.0,
    band_low: float = 0.1,
    band_high: float = 75.0,
    notch_freq: float = 50.0,
    clip_labels: Optional[Sequence[int]] = None,
    overwrite: bool = False,
) -> str:
    """Preprocess every SEED ``.cnt`` into per-clip npy trial directories.

    Parameters
    ----------
    dataset_path : str
        Path to the SEED root (containing ``Chinese/01-EEG-raw/``).
    output_root : str, optional
        Where to write the preprocessed output.  Defaults to
        ``<dataset_path>/Chinese/preprocessed``.
    target_sfreq : float
        Target sampling frequency after resampling.
    clip_labels : sequence of int, optional
        Per-clip emotion labels (length 15).  Defaults to the standard
        SEED label sequence.
    overwrite : bool
        Re-process trials that already exist on disk.

    Returns
    -------
    str
        The *output_root* path.
    """
    eeg_raw_dir = os.path.join(dataset_path, "Chinese", "01-EEG-raw")
    if not os.path.isdir(eeg_raw_dir):
        raise FileNotFoundError(
            f"Expected raw EEG directory at {eeg_raw_dir}"
        )

    time_txt = os.path.join(eeg_raw_dir, "time.txt")
    boundaries = _parse_time_boundaries(time_txt)
    n_clips = len(boundaries)

    labels = list(clip_labels) if clip_labels is not None else list(SEED_CLIP_LABELS)
    if len(labels) != n_clips:
        raise ValueError(
            f"clip_labels length ({len(labels)}) != clip count ({n_clips})"
        )

    if output_root is None:
        output_root = os.path.join(dataset_path, "Chinese", "preprocessed")
    os.makedirs(output_root, exist_ok=True)

    cnt_files = sorted(
        f for f in os.listdir(eeg_raw_dir) if f.lower().endswith(".cnt")
    )
    if not cnt_files:
        raise FileNotFoundError(f"No .cnt files found in {eeg_raw_dir}")

    n_ok, n_skip, n_err = 0, 0, 0

    for cnt_name in cnt_files:
        cnt_path = os.path.join(eeg_raw_dir, cnt_name)
        base = cnt_name[:-4]  # e.g. "1_1"
        parts = base.split("_")
        if len(parts) != 2:
            warnings.warn(f"Unexpected CNT filename pattern: {cnt_name}")
            continue
        subject_id, session_id = parts[0], int(parts[1])

        print(f"[seed] Processing {cnt_name} (subject={subject_id}, "
              f"session={session_id}) ...")

        try:
            eeg_full, ts_full, cnt_info = _preprocess_cnt(
                cnt_path, target_sfreq, band_low, band_high, notch_freq,
            )
        except Exception as exc:
            warnings.warn(f"Skipping {cnt_path}: {exc}")
            n_err += 1
            continue

        original_sfreq = cnt_info["original_sfreq"]

        for clip_idx, (start_1k, end_1k) in enumerate(boundaries):
            clip_num = clip_idx + 1
            start_rs = int(round(start_1k * target_sfreq / original_sfreq))
            end_rs = int(round(end_1k * target_sfreq / original_sfreq))

            start_rs = max(0, min(start_rs, eeg_full.shape[0]))
            end_rs = max(start_rs, min(end_rs, eeg_full.shape[0]))
            if end_rs - start_rs < 1:
                warnings.warn(
                    f"Empty segment for clip {clip_num} in {cnt_name}"
                )
                continue

            trial_id = f"sub{subject_id}_sess{session_id}_clip{clip_num:02d}"
            out_dir = os.path.join(output_root, trial_id)

            if (
                not overwrite
                and os.path.isfile(os.path.join(out_dir, "meta.json"))
            ):
                n_skip += 1
                continue

            eeg_clip = eeg_full[start_rs:end_rs]
            ts_clip = ts_full[start_rs:end_rs]

            original_label = labels[clip_idx]
            classifier_label = original_label + 1  # → 0/1/2

            meta = {
                **cnt_info,
                "trial_id": trial_id,
                "subject_id": subject_id,
                "session_id": session_id,
                "clip_id": clip_num,
                "label": classifier_label,
                "original_label": original_label,
                "label_name": SEED_LABEL_NAMES.get(original_label, "unknown"),
                "num_samples": int(eeg_clip.shape[0]),
                "clip_start_sample_1kHz": start_1k,
                "clip_end_sample_1kHz": end_1k,
            }

            _write_trial(out_dir, eeg_clip, ts_clip, meta, overwrite=overwrite)
            n_ok += 1

        if (n_ok + n_err) % 30 == 0:
            print(f"  progress: {n_ok} ok, {n_err} err, {n_skip} skip ...")

    print(
        f"[seed] Done: {n_ok} clips written, {n_err} errors, "
        f"{n_skip} skipped → {output_root}"
    )
    return output_root


# ===================================================================
# Split generation
# ===================================================================

def generate_seed_splits(
    preprocessed_path: str,
    mode: str = "subject_independent",
    val_ratio: float = 0.2,
    seed: int = 42,
    val_subjects: Optional[Sequence[str]] = None,
    train_sessions: Optional[Sequence[int]] = None,
    val_sessions: Optional[Sequence[int]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Create a ``splits.json`` for the SEED preprocessed directory.

    Parameters
    ----------
    preprocessed_path : str
        Path containing the trial directories (output of
        :func:`preprocess_seed_to_npy`).
    mode : str
        ``"subject_independent"`` — split by subject, or
        ``"subject_dependent"``  — split by session.
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
    # Discover all subjects and sessions from meta files
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
        t_sess = list(train_sessions) if train_sessions else [1, 2]
        v_sess = list(val_sessions) if val_sessions else [3]
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
    print(f"[seed] Splits written: {output_path}")
    return output_path


# ===================================================================
# Shared helper for video-per-class splits (all SEED-family datasets)
# ===================================================================

def _generate_video_per_class_splits_impl(
    preprocessed_path: str,
    independent_splits_path: str,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> str:
    """Core logic for generating video-per-class splits.

    Reads *val_subjects* from an existing subject-independent split JSON,
    then for those subjects selects exactly one clip per class for training
    and keeps the rest for validation.

    Works for any SEED-family dataset (SEED, SEED-IV, SEED-V, SEED-VII)
    because they all share the same ``meta.json`` structure.

    Parameters
    ----------
    preprocessed_path : str
        Path containing the trial directories.
    independent_splits_path : str
        Path to an existing ``splits_subject_independent.json``.
    seed : int
        Random seed for reproducible clip selection.
    output_path : str, optional
        Where to write the JSON.  Defaults to
        ``<preprocessed_path>/splits_video_per_class.json``.

    Returns
    -------
    str
        Path to the written JSON.
    """
    from .seed import _iter_trial_dirs

    with open(independent_splits_path, "r", encoding="utf-8") as f:
        indep = json.load(f)
    val_subjects = set(str(s) for s in indep["val_subjects"])

    all_trials: List[Tuple[str, int]] = []
    for _, meta in _iter_trial_dirs(preprocessed_path, subjects=val_subjects):
        all_trials.append((meta["trial_id"], int(meta["label"])))

    by_label: Dict[int, List[str]] = defaultdict(list)
    for trial_id, label in all_trials:
        by_label[label].append(trial_id)

    missing = [lbl for lbl, ids in by_label.items() if len(ids) == 0]
    if missing:
        raise ValueError(
            f"No clips found for label(s) {missing} in {preprocessed_path}"
        )
    if not by_label:
        raise ValueError(
            f"No clips found for val_subjects {val_subjects} in {preprocessed_path}"
        )

    rng = np.random.RandomState(seed)
    train_trial_ids: List[str] = []
    for label in sorted(by_label):
        candidates = sorted(by_label[label])
        train_trial_ids.append(str(rng.choice(candidates)))

    train_set = set(train_trial_ids)
    val_trial_ids = sorted(tid for tid, _ in all_trials if tid not in train_set)

    if output_path is None:
        output_path = os.path.join(preprocessed_path, "splits_video_per_class.json")

    splits = {
        "mode": "video_per_class",
        "seed": seed,
        "source_independent_splits": independent_splits_path,
        "val_subjects": sorted(val_subjects, key=lambda s: int(s)),
        "n_train_clips": len(train_trial_ids),
        "n_val_clips": len(val_trial_ids),
        "train_trial_ids": sorted(train_trial_ids),
        "val_trial_ids": val_trial_ids,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    return output_path


def generate_seed_video_per_class_splits(
    preprocessed_path: str,
    independent_splits_path: str,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> str:
    """Generate a ``splits_video_per_class.json`` for the SEED dataset.

    Uses the validation subjects from *independent_splits_path* and
    selects one clip per class for training; all remaining clips from
    those subjects form the validation set.

    Parameters
    ----------
    preprocessed_path : str
        Path to ``/data/seeds/seed/Chinese/preprocessed``.
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
    print(f"[seed] Video-per-class splits written: {path}")
    return path
