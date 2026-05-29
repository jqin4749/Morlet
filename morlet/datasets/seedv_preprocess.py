from __future__ import annotations

import json
import os
import re
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .seed_preprocess import (
    SEED_STANDARD_CHANNELS,
    _read_cnt,
    _preprocess_cnt,
    _write_trial,
    _generate_video_per_class_splits_impl,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDV_LABEL_NAMES: Dict[int, str] = {
    0: "disgust",
    1: "fear",
    2: "sad",
    3: "neutral",
    4: "happy",
}

NUM_CLASSES = 5

# Per-session clip emotion labels (derived from emotion_label_and_stimuli_order.xlsx).
# Session 1 has a unique order; sessions 2 and 3 share the same order.
SEEDV_SESSION_LABELS: Dict[int, List[int]] = {
    1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}


# ---------------------------------------------------------------------------
# Time boundary parsing  (seconds, per-session)
# ---------------------------------------------------------------------------

def _parse_seedv_time_boundaries(
    txt_path: str,
) -> Dict[int, List[Tuple[int, int]]]:
    """Parse ``trial_start_end_timestamp.txt`` for SEED-V.

    Returns a dict mapping session id (1/2/3) to a list of
    ``(start_second, end_second)`` tuples.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    boundaries: Dict[int, List[Tuple[int, int]]] = {}
    current_session: Optional[int] = None

    starts: Optional[List[int]] = None
    ends: Optional[List[int]] = None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        sess_m = re.match(r"Session\s+(\d+)\s*:", line)
        if sess_m:
            current_session = int(sess_m.group(1))
            starts = None
            ends = None
            continue

        if current_session is None:
            continue

        start_m = re.match(r"start_second\s*:\s*\[([^\]]+)\]", line)
        if start_m:
            starts = [int(x.strip()) for x in start_m.group(1).split(",") if x.strip()]
            if starts is not None and ends is not None:
                boundaries[current_session] = list(zip(starts, ends))
            continue

        end_m = re.match(r"end_second\s*:\s*\[([^\]]+)\]", line)
        if end_m:
            ends = [int(x.strip()) for x in end_m.group(1).split(",") if x.strip()]
            if starts is not None and ends is not None:
                boundaries[current_session] = list(zip(starts, ends))
            continue

    if not boundaries:
        raise ValueError(
            f"Could not parse any session boundaries from {txt_path}"
        )
    return boundaries


# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------

def _parse_cnt_filename(cnt_name: str) -> Optional[Tuple[str, int]]:
    """Extract ``(subject_id, session_id)`` from a SEED-V CNT filename.

    Expected patterns:
        ``subjectID_sessionID_date.cnt``         → 3 parts
        ``subjectID_sessionID_date_repaired.cnt`` → 4 parts (repaired)

    Returns *None* for unrecognised patterns.
    """
    base = cnt_name[:-4] if cnt_name.lower().endswith(".cnt") else cnt_name
    parts = base.split("_")
    if len(parts) == 3:
        # e.g. "1_1_20180804"
        return parts[0], int(parts[1])
    if len(parts) == 4 and parts[3].lower() == "repaired":
        # e.g. "7_1_20180411_repaired"
        return parts[0], int(parts[1])
    return None


def _collect_cnt_files(eeg_raw_dir: str) -> List[Tuple[str, str, int]]:
    """Collect CNT files, preferring ``_repaired`` versions when both exist.

    Returns a list of ``(cnt_filename, subject_id, session_id)``, sorted
    by ``(subject_id, session_id)``.
    """
    raw_files: Dict[Tuple[str, int], str] = {}

    for f in sorted(os.listdir(eeg_raw_dir)):
        if not f.lower().endswith(".cnt"):
            continue
        parsed = _parse_cnt_filename(f)
        if parsed is None:
            warnings.warn(f"Skipping unrecognised CNT filename: {f}")
            continue
        subject_id, session_id = parsed
        key = (subject_id, session_id)

        is_repaired = "_repaired" in f.lower()
        if key in raw_files:
            existing_is_repaired = "_repaired" in raw_files[key].lower()
            if is_repaired and not existing_is_repaired:
                raw_files[key] = f
        else:
            raw_files[key] = f

    results = [
        (fname, subj, sess)
        for (subj, sess), fname in raw_files.items()
    ]
    results.sort(key=lambda t: (int(t[1]), t[2]))
    return results


# ===================================================================
# Full preprocessing: SEED-V raw CNT → per-clip npy
# ===================================================================

def preprocess_seedv_to_npy(
    dataset_path: str,
    output_root: Optional[str] = None,
    target_sfreq: float = 200.0,
    band_low: float = 0.1,
    band_high: float = 75.0,
    notch_freq: float = 50.0,
    overwrite: bool = False,
) -> str:
    """Preprocess every SEED-V ``.cnt`` into per-clip npy trial directories.

    Parameters
    ----------
    dataset_path : str
        Path to the SEED-V root (containing ``EEG_raw/`` and
        ``trial_start_end_timestamp.txt``).
    output_root : str, optional
        Where to write the preprocessed output.  Defaults to
        ``<dataset_path>/preprocessed``.
    target_sfreq : float
        Target sampling frequency after resampling.
    band_low / band_high : float
        Band-pass filter edges in Hz.
    notch_freq : float
        Power-line notch frequency in Hz.
    overwrite : bool
        Re-process trials that already exist on disk.

    Returns
    -------
    str
        The *output_root* path.
    """
    eeg_raw_dir = os.path.join(dataset_path, "EEG_raw")
    if not os.path.isdir(eeg_raw_dir):
        raise FileNotFoundError(
            f"Expected raw EEG directory at {eeg_raw_dir}"
        )

    time_txt = os.path.join(dataset_path, "trial_start_end_timestamp.txt")
    session_boundaries = _parse_seedv_time_boundaries(time_txt)

    if output_root is None:
        output_root = os.path.join(dataset_path, "preprocessed")
    os.makedirs(output_root, exist_ok=True)

    cnt_entries = _collect_cnt_files(eeg_raw_dir)
    if not cnt_entries:
        raise FileNotFoundError(f"No valid .cnt files found in {eeg_raw_dir}")

    n_ok, n_skip, n_err = 0, 0, 0

    for cnt_name, subject_id, session_id in cnt_entries:
        cnt_path = os.path.join(eeg_raw_dir, cnt_name)

        if session_id not in session_boundaries:
            warnings.warn(
                f"No time boundaries for session {session_id}, "
                f"skipping {cnt_name}"
            )
            n_err += 1
            continue

        boundaries = session_boundaries[session_id]
        labels = SEEDV_SESSION_LABELS.get(session_id)
        if labels is None:
            warnings.warn(
                f"No label sequence for session {session_id}, "
                f"skipping {cnt_name}"
            )
            n_err += 1
            continue

        if len(labels) != len(boundaries):
            warnings.warn(
                f"Label count ({len(labels)}) != boundary count "
                f"({len(boundaries)}) for session {session_id}"
            )
            n_err += 1
            continue

        print(
            f"[seedv] Processing {cnt_name} "
            f"(subject={subject_id}, session={session_id}) ..."
        )

        try:
            eeg_full, ts_full, cnt_info = _preprocess_cnt(
                cnt_path, target_sfreq, band_low, band_high, notch_freq,
            )
        except Exception as exc:
            warnings.warn(f"Skipping {cnt_path}: {exc}")
            n_err += 1
            continue

        for clip_idx, (start_sec, end_sec) in enumerate(boundaries):
            clip_num = clip_idx + 1

            start_rs = int(round(start_sec * target_sfreq))
            end_rs = int(round(end_sec * target_sfreq))

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

            label = labels[clip_idx]

            meta = {
                **cnt_info,
                "trial_id": trial_id,
                "subject_id": subject_id,
                "session_id": session_id,
                "clip_id": clip_num,
                "label": label,
                "label_name": SEEDV_LABEL_NAMES.get(label, "unknown"),
                "num_samples": int(eeg_clip.shape[0]),
                "clip_start_second": start_sec,
                "clip_end_second": end_sec,
            }

            _write_trial(out_dir, eeg_clip, ts_clip, meta, overwrite=overwrite)
            n_ok += 1

        if (n_ok + n_err) % 30 == 0:
            print(f"  progress: {n_ok} ok, {n_err} err, {n_skip} skip ...")

    print(
        f"[seedv] Done: {n_ok} clips written, {n_err} errors, "
        f"{n_skip} skipped → {output_root}"
    )
    return output_root


# ===================================================================
# Split generation
# ===================================================================

def generate_seedv_splits(
    preprocessed_path: str,
    mode: str = "subject_independent",
    val_ratio: float = 0.2,
    seed: int = 42,
    val_subjects: Optional[Sequence[str]] = None,
    train_sessions: Optional[Sequence[int]] = None,
    val_sessions: Optional[Sequence[int]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Create a ``splits.json`` for the SEED-V preprocessed directory.

    Parameters
    ----------
    preprocessed_path : str
        Path containing the trial directories (output of
        :func:`preprocess_seedv_to_npy`).
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
    print(f"[seedv] Splits written: {output_path}")
    return output_path


def generate_seedv_video_per_class_splits(
    preprocessed_path: str,
    independent_splits_path: str,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> str:
    """Generate a ``splits_video_per_class.json`` for the SEED-V dataset.

    Uses the validation subjects from *independent_splits_path* and
    selects one clip per class for training; all remaining clips from
    those subjects form the validation set.

    Parameters
    ----------
    preprocessed_path : str
        Path to ``/data/seeds/seed-v/preprocessed``.
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
    print(f"[seedv] Video-per-class splits written: {path}")
    return path
