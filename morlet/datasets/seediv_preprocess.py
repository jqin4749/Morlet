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

SEEDIV_LABEL_NAMES: Dict[int, str] = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}

NUM_CLASSES = 4

# Per-session clip labels from ReadMe.txt in /data/seeds/seed-iv
SEEDIV_SESSION_LABELS: Dict[int, List[int]] = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

SAMPLING_FREQUENCY = 200.0
TRIALS_PER_SESSION = 24


def _load_seediv_channel_order(dataset_path: str) -> List[str]:
    """Load channel order from ``channel_62_pos.locs`` and return names.

    The file has 4 columns: index, azimuth, radius, label.
    """
    locs_path = os.path.join(dataset_path, "channel_62_pos.locs")
    if not os.path.isfile(locs_path):
        raise FileNotFoundError(
            f"Expected channel position file at {locs_path}"
        )

    ch_names: List[str] = []
    with open(locs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            name = parts[3].upper()
            ch_names.append(name)

    if len(ch_names) != len(SEED_STANDARD_CHANNELS):
        warnings.warn(
            f"channel_62_pos.locs has {len(ch_names)} entries, "
            f"expected {len(SEED_STANDARD_CHANNELS)}"
        )
    return ch_names


def _build_reorder_indices(
    local_channels: Sequence[str],
) -> List[int]:
    """Map local SEED-IV order to SEED_STANDARD_CHANNELS order.

    Returns indices ``idx`` such that::

        eeg_reordered = eeg_local[idx, :]

    yields channels in ``SEED_STANDARD_CHANNELS`` order.
    """
    name_to_idx = {ch.upper(): i for i, ch in enumerate(local_channels)}
    indices: List[int] = []
    missing: List[str] = []
    for ch in SEED_STANDARD_CHANNELS:
        idx = name_to_idx.get(ch)
        if idx is None:
            missing.append(ch)
        else:
            indices.append(idx)
    if missing:
        raise ValueError(
            f"Missing standard EEG channels in SEED-IV data: {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
    return indices


def _parse_subject_id(mat_name: str) -> str:
    """Extract numeric subject id from a SEED-IV mat filename.

    Filenames follow ``{SubjectName}_{Date}.mat``, where SubjectName is
    typically ``sub1``, ``sub2``, ... We store the numeric id as string,
    e.g. ``\"1\"``, consistent with other SEED-family preprocessors.
    """
    base = mat_name[:-4] if mat_name.lower().endswith(".mat") else mat_name
    if "_" in base:
        subject_part = base.split("_", 1)[0]
    else:
        subject_part = base
    sp = subject_part.lower()
    if sp.startswith("sub"):
        sp = sp[3:]
    return sp


def preprocess_seediv_to_npy(
    dataset_path: str,
    output_root: Optional[str] = None,
    target_sfreq: float = SAMPLING_FREQUENCY,
    band_low: float = 1.0,
    band_high: float = 75.0,
    notch_freq: float = 50.0,
    overwrite: bool = False,
) -> str:
    """Convert SEED-IV EEG ``.mat`` files into per-trial npy directories.

    Parameters
    ----------
    dataset_path : str
        Path to the SEED-IV root (containing ``eeg_raw_data/`` and
        ``channel_62_pos.locs``).
    output_root : str, optional
        Where to write the preprocessed output.  Defaults to
        ``<dataset_path>/preprocessed``.
    target_sfreq : float
        Target sampling frequency after (optional) resampling.  Default
        is 200 Hz, consistent with other SEED-family preprocessors.
    band_low / band_high : float
        Band-pass filter edges in Hz.
    notch_freq : float
        Power-line notch frequency in Hz.
    overwrite : bool
        Re-process trials that already exist on disk.

    Notes
    -----
    According to the SEED-IV description
    (\"Feature Extraction\" section in the dataset page), raw EEG is
    downsampled to 200 Hz and bandpass filtered between 1–75 Hz before
    PSD/DE computation. To keep the preprocessing pipeline consistent
    with other SEED-family scripts in this repository, we explicitly
    apply a 1–75 Hz band-pass filter and 50 Hz notch filter here via
    MNE, and (optionally) resample to *target_sfreq* (default 200 Hz).
    """
    import scipy.io
    import mne

    mne.set_log_level("ERROR")

    eeg_root = os.path.join(dataset_path, "eeg_raw_data")
    if not os.path.isdir(eeg_root):
        raise FileNotFoundError(
            f"Expected SEED-IV eeg_raw_data directory at {eeg_root}"
        )

    if output_root is None:
        output_root = os.path.join(dataset_path, "preprocessed")
    os.makedirs(output_root, exist_ok=True)

    local_channels = _load_seediv_channel_order(dataset_path)
    reorder_idx = _build_reorder_indices(local_channels)

    n_ok, n_skip, n_err = 0, 0, 0

    for sess_name in sorted(os.listdir(eeg_root)):
        sess_dir = os.path.join(eeg_root, sess_name)
        if not os.path.isdir(sess_dir):
            continue
        try:
            session_id = int(sess_name)
        except ValueError:
            warnings.warn(f"Skipping non-session directory: {sess_dir}")
            continue

        if session_id not in SEEDIV_SESSION_LABELS:
            warnings.warn(
                f"No label sequence for session {session_id}, skipping"
            )
            continue

        labels = SEEDIV_SESSION_LABELS[session_id]
        if len(labels) != TRIALS_PER_SESSION:
            warnings.warn(
                f"Label count ({len(labels)}) != {TRIALS_PER_SESSION} "
                f"for session {session_id}"
            )

        mat_files = sorted(
            f for f in os.listdir(sess_dir) if f.lower().endswith(".mat")
        )
        if not mat_files:
            warnings.warn(f"No .mat files found in {sess_dir}")
            continue

        for mat_name in mat_files:
            mat_path = os.path.join(sess_dir, mat_name)
            subject_id = _parse_subject_id(mat_name)

            print(
                f"[seediv] Processing {mat_name} "
                f"(subject={subject_id}, session={session_id}) ..."
            )

            try:
                mat_data = scipy.io.loadmat(mat_path)
            except Exception as exc:
                warnings.warn(f"Skipping {mat_path}: {exc}")
                n_err += 1
                continue

            # ------------------------------------------------------------------
            # Infer per-subject field naming pattern, e.g. "tyc_eeg1" .. "tyc_eeg24"
            # instead of assuming a fixed "cz_eeg*" prefix.
            #
            # NOTE: Use a simple ".*eeg(\\d+)$" pattern to avoid relying on a
            # specific prefix / underscore layout.
            # ------------------------------------------------------------------
            import re as _re

            field_map: Dict[int, str] = {}
            for key in mat_data.keys():
                if key.startswith("__"):
                    continue
                m = _re.match(r".*eeg(\d+)$", str(key))
                if not m:
                    continue
                idx = int(m.group(1))
                field_map[idx] = key

            missing_indices = [i for i in range(1, TRIALS_PER_SESSION + 1) if i not in field_map]
            if missing_indices:
                warnings.warn(
                    f"Missing EEG fields {missing_indices} in {mat_name}; "
                    f"only found indices {sorted(field_map.keys())}"
                )

            for clip_idx in range(1, TRIALS_PER_SESSION + 1):
                field_name = field_map.get(clip_idx)
                if field_name is None:
                    # Skip this clip if the corresponding EEG field is absent
                    n_err += 1
                    continue

                eeg_raw = mat_data[field_name]
                if eeg_raw.ndim != 2:
                    warnings.warn(
                        f"Unexpected shape {eeg_raw.shape} for {field_name} "
                        f"in {mat_name}, expected (channels, time)"
                    )
                    n_err += 1
                    continue

                n_channels, n_samples = eeg_raw.shape
                if n_channels != len(local_channels):
                    warnings.warn(
                        f"Channel count {n_channels} != "
                        f"{len(local_channels)} in {mat_name}/{field_name}"
                    )

                eeg_local = eeg_raw.astype(np.float32)
                eeg_reordered = eeg_local[reorder_idx, :]  # (62, time)

                # ------------------------------------------------------------------
                # Filtering and (optional) resampling using MNE, mirroring the
                # SEED / SEED-V pipeline (1–75 Hz band-pass + 50 Hz notch).
                # ------------------------------------------------------------------
                original_sfreq = SAMPLING_FREQUENCY
                info = mne.create_info(
                    ch_names=list(SEED_STANDARD_CHANNELS),
                    sfreq=original_sfreq,
                    ch_types="eeg",
                )
                # Treat values as micro-volts and convert to volts for MNE
                raw = mne.io.RawArray(eeg_reordered * 1e-6, info)

                raw.filter(l_freq=band_low, h_freq=band_high)
                raw.notch_filter(notch_freq)

                if float(target_sfreq) != float(original_sfreq):
                    raw.resample(target_sfreq)

                try:
                    eeg = raw.get_data(units="uV").T.astype(np.float32)
                except TypeError:
                    eeg = (raw.get_data() * 1e6).T.astype(np.float32)
                timestamps = raw.times.astype(np.float32)
                raw.close()

                if 0 <= clip_idx - 1 < len(labels):
                    label = int(labels[clip_idx - 1])
                else:
                    warnings.warn(
                        f"No label for clip {clip_idx} in session "
                        f"{session_id}, defaulting to 0"
                    )
                    label = 0

                trial_id = (
                    f"sub{subject_id}_sess{session_id}_clip{clip_idx:02d}"
                )
                out_dir = os.path.join(output_root, trial_id)

                if (
                    not overwrite
                    and os.path.isfile(os.path.join(out_dir, "meta.json"))
                ):
                    n_skip += 1
                    continue

                meta = {
                    "trial_id": trial_id,
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "clip_id": clip_idx,
                    "label": label,
                    "label_name": SEEDIV_LABEL_NAMES.get(label, "unknown"),
                    "num_samples": int(eeg.shape[0]),
                    "num_eeg_channels": int(eeg.shape[1]),
                    "sampling_frequency": float(target_sfreq),
                    "channel_labels": list(SEED_STANDARD_CHANNELS),
                    "bandpass_hz": [float(band_low), float(band_high)],
                    "notch_hz": float(notch_freq),
                }

                _write_trial(out_dir, eeg, timestamps, meta, overwrite=overwrite)
                n_ok += 1

            if (n_ok + n_err) % 100 == 0 and n_ok + n_err > 0:
                print(
                    f"  progress: {n_ok} ok, {n_err} err, {n_skip} skip ..."
                )

    print(
        f"[seediv] Done: {n_ok} trials written, {n_err} errors, "
        f"{n_skip} skipped -> {output_root}"
    )
    return output_root


# ===================================================================
# Split generation
# ===================================================================

def generate_seediv_splits(
    preprocessed_path: str,
    mode: str = "subject_independent",
    val_ratio: float = 0.2,
    seed: int = 42,
    val_subjects: Optional[Sequence[str]] = None,
    train_sessions: Optional[Sequence[int]] = None,
    val_sessions: Optional[Sequence[int]] = None,
    output_path: Optional[str] = None,
) -> str:
    """Create a ``splits.json`` for the SEED-IV preprocessed directory.

    Parameters
    ----------
    preprocessed_path : str
        Path containing the trial directories (output of
        :func:`preprocess_seediv_to_npy`).
    mode : str
        ``\"subject_independent\"`` — split by subject, or
        ``\"subject_dependent\"``  — split by session.
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
    import numpy as np

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
            "val_ratio": val_ratio,
            "n_subjects": len(subjects_sorted),
            "train_subjects": subjects_sorted,
            "val_subjects": subjects_sorted,
            "train_sessions": sorted(t_sess),
            "val_sessions": sorted(v_sess),
        }

    else:
        raise ValueError(
            f"Unsupported mode '{mode}', "
            "expected 'subject_independent' or 'subject_dependent'"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)
    return output_path


def generate_seediv_video_per_class_splits(
    preprocessed_path: str,
    independent_splits_path: str,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> str:
    """Generate a ``splits_video_per_class.json`` for the SEED-IV dataset.

    Uses the validation subjects from *independent_splits_path* and
    selects one clip per class for training; all remaining clips from
    those subjects form the validation set.

    Parameters
    ----------
    preprocessed_path : str
        Path to ``/data/seeds/seed-iv/preprocessed``.
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
    print(f"[seediv] Video-per-class splits written: {path}")
    return path

