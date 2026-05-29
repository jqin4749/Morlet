from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


_NORM_MODES = frozenset(("window", "trial", "subject", "none"))


def _load_subject_stats_npz(
    subject_stats_dir: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load per-subject ``mean`` / ``std`` vectors from ``*.npz`` in *subject_stats_dir*."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if not os.path.isdir(subject_stats_dir):
        return out
    for fn in sorted(os.listdir(subject_stats_dir)):
        if not fn.endswith(".npz"):
            continue
        path = os.path.join(subject_stats_dir, fn)
        z = np.load(path, allow_pickle=True)
        if "subject_id" in z.files:
            sid_raw = z["subject_id"]
            sid = str(sid_raw.item() if hasattr(sid_raw, "item") else sid_raw[()])
        else:
            sid = os.path.splitext(fn)[0]
        out[str(sid)] = (
            np.asarray(z["mean"], dtype=np.float32),
            np.asarray(z["std"], dtype=np.float32),
        )
    return out
import torch
from torch.utils.data import Dataset


# ===================================================================
# Helpers
# ===================================================================

def _iter_trial_dirs(
    preprocessed_path: str,
    subjects: Optional[set] = None,
    sessions: Optional[set] = None,
    trial_ids: Optional[set] = None,
) -> Iterable[Tuple[str, dict]]:
    """Yield ``(trial_dir, meta_dict)`` for valid preprocessed trial dirs.

    Supports filtering by *subjects* (set of str subject IDs) and/or
    *sessions* (set of int session IDs).
    """
    for name in sorted(os.listdir(preprocessed_path)):
        trial_dir = os.path.join(preprocessed_path, name)
        meta_path = os.path.join(trial_dir, "meta.json")
        eeg_path = os.path.join(trial_dir, "eeg.npy")
        if not (
            os.path.isdir(trial_dir)
            and os.path.isfile(meta_path)
            and os.path.isfile(eeg_path)
        ):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if subjects is not None:
            subj = str(meta.get("subject_id", ""))
            if subj not in subjects:
                continue
        if sessions is not None:
            sess = meta.get("session_id")
            if sess is not None and int(sess) not in sessions:
                continue
        if trial_ids is not None:
            if meta.get("trial_id", "") not in trial_ids:
                continue
        yield trial_dir, meta


# ===================================================================
# Dataclasses
# ===================================================================

@dataclass
class _TrialInfo:
    name: str
    subject_id: str
    eeg_path: str
    timestamps_path: str
    mean_path: str
    std_path: str
    meta_path: str
    meta: dict
    start_idx: int
    end_idx: int
    n_samples: int
    n_channels: int


@dataclass
class _TrialCache:
    eeg: np.ndarray         # memmap (N, C)
    timestamps: np.ndarray   # memmap (N,)
    mean: np.ndarray         # (C,)
    std: np.ndarray          # (C,)


# ===================================================================
# Base class
# ===================================================================

class _BaseSeedDataset(Dataset):
    """Scanning / caching / windowing logic for preprocessed SEED data."""

    def __init__(
        self,
        dataset_path: str,
        interval_length: int,
        subjects: Optional[Sequence[str]] = None,
        sessions: Optional[Sequence[int]] = None,
        trial_ids: Optional[Sequence[str]] = None,
        cache_size: int = 10,
        stride: Optional[int] = None,
        padding: int = 0,
        normalize: str = "window",
        subject_stats_dir: Optional[str] = None,
        freq_baseline_n_segments: int = 0,
    ) -> None:
        self.freq_baseline_n_segments = max(0, int(freq_baseline_n_segments))

        interval_length = int(interval_length)
        if interval_length <= 0:
            raise ValueError("interval_length must be positive")
        padding = int(padding)
        if padding < 0:
            raise ValueError("padding must be non-negative")

        norm = str(normalize).lower()
        if norm not in _NORM_MODES:
            raise ValueError(
                f"normalize must be one of {sorted(_NORM_MODES)}, got {normalize!r}",
            )
        self.normalize = norm
        self.subject_stats_dir = (
            subject_stats_dir if subject_stats_dir is not None
            else os.path.join(dataset_path, "subject_stats")
        )
        self._subject_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if self.normalize == "subject":
            self._subject_stats = _load_subject_stats_npz(self.subject_stats_dir)
            if not self._subject_stats:
                warnings.warn(
                    f"normalize='subject' but no stats found under {self.subject_stats_dir!r}; "
                    "falling back to 'trial' normalization.",
                )
                self.normalize = "trial"

        self.preprocessed_path = dataset_path
        self.window_size = interval_length
        self.padding = padding
        self.subjects = set(str(s) for s in subjects) if subjects is not None else None
        self.sessions = set(int(s) for s in sessions) if sessions is not None else None
        self.trial_ids = set(str(t) for t in trial_ids) if trial_ids is not None else None
        self.cache_size = max(1, int(cache_size))

        if stride is None:
            self.stride_size = self.window_size
        else:
            stride = int(stride)
            if stride <= 0:
                raise ValueError("stride must be positive")
            if stride > interval_length:
                raise ValueError("stride must not exceed interval_length")
            self.stride_size = stride

        self.trial_infos: List[_TrialInfo] = []
        self.index: List[Any] = []
        self._trial_cache: Dict[int, _TrialCache] = {}
        self._cache_order: List[int] = []
        self._expected_n_channels: Optional[int] = None

        self._scan_trials()

    # ----- scanning -----

    def _scan_trials(self) -> None:
        for trial_dir, meta in _iter_trial_dirs(
            self.preprocessed_path, self.subjects, self.sessions, self.trial_ids,
        ):
            eeg_path = os.path.join(trial_dir, "eeg.npy")
            ts_path = os.path.join(trial_dir, "timestamps.npy")
            mean_path = os.path.join(trial_dir, "mean.npy")
            std_path = os.path.join(trial_dir, "std.npy")
            meta_path = os.path.join(trial_dir, "meta.json")

            if not all(os.path.isfile(p) for p in (ts_path, mean_path, std_path)):
                continue

            eeg_tmp = np.load(eeg_path, mmap_mode="r")
            n_samples, n_channels = eeg_tmp.shape
            if n_samples < self.window_size:
                continue
            if self._expected_n_channels is None:
                self._expected_n_channels = int(n_channels)
            elif int(n_channels) != self._expected_n_channels:
                warnings.warn(
                    f"Skipping trial '{os.path.basename(trial_dir)}': "
                    f"n_channels={n_channels} != expected={self._expected_n_channels}"
                )
                continue

            trial_idx = len(self.trial_infos)
            info = _TrialInfo(
                name=os.path.basename(trial_dir),
                subject_id=str(meta.get("subject_id", "")),
                eeg_path=eeg_path,
                timestamps_path=ts_path,
                mean_path=mean_path,
                std_path=std_path,
                meta_path=meta_path,
                meta=meta,
                start_idx=0,
                end_idx=n_samples,
                n_samples=n_samples,
                n_channels=n_channels,
            )
            self.trial_infos.append(info)
            self._build_index(trial_idx, info)

    def _build_index(self, trial_idx: int, info: _TrialInfo) -> None:
        raise NotImplementedError

    # ----- loading & LRU cache -----

    def _load_trial_data(self, trial_idx: int) -> _TrialCache:
        info = self.trial_infos[trial_idx]
        return _TrialCache(
            eeg=np.load(info.eeg_path, mmap_mode="r"),
            timestamps=np.load(info.timestamps_path, mmap_mode="r"),
            mean=np.load(info.mean_path).astype(np.float32),
            std=np.load(info.std_path).astype(np.float32),
        )

    def _get_trial(self, trial_idx: int) -> _TrialCache:
        if trial_idx in self._trial_cache:
            self._cache_order.remove(trial_idx)
            self._cache_order.append(trial_idx)
            return self._trial_cache[trial_idx]
        trial = self._load_trial_data(trial_idx)
        self._trial_cache[trial_idx] = trial
        self._cache_order.append(trial_idx)
        while len(self._cache_order) > self.cache_size:
            evict = self._cache_order.pop(0)
            self._trial_cache.pop(evict, None)
        return trial

    def _norm_mean_std_for_trial(
        self, trial: _TrialCache, trial_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) vectors (C,) for *trial* / *subject* normalization."""
        info = self.trial_infos[trial_idx]
        if self.normalize == "trial":
            return trial.mean, trial.std
        sid = info.subject_id
        if sid in self._subject_stats:
            m, s = self._subject_stats[sid]
            return m, s
        warnings.warn(
            f"No subject_stats entry for subject_id={sid!r} (trial {info.name}); "
            "using trial-level mean/std.",
        )
        return trial.mean, trial.std

    def _zero_eeg_window_tensor(self, channel_last: bool, trial_idx: int) -> torch.Tensor:
        c_ch = int(self.trial_infos[trial_idx].n_channels)
        t_w = self.window_size
        if channel_last:
            return torch.zeros(t_w, c_ch, dtype=torch.float32)
        return torch.zeros(c_ch, t_w, dtype=torch.float32)

    def _eeg_history_stack(
        self,
        trial: _TrialCache,
        start: int,
        trial_idx: int,
        channel_last: bool,
    ) -> torch.Tensor:
        """Stack neighboring segments (prev n + next n, excluding self).

        Returns shape ``[2n, T, C]`` or ``[2n, C, T]`` with the same normalization
        as the current window.
        """
        n = self.freq_baseline_n_segments
        info = self.trial_infos[trial_idx]
        slots: List[torch.Tensor] = []
        template: Optional[torch.Tensor] = None
        candidate_starts: List[int] = []
        for j in range(1, n + 1):
            prev = start - j * self.stride_size
            candidate_starts.append(prev)
        for j in range(1, n + 1):
            nxt = start + j * self.stride_size
            candidate_starts.append(nxt)
        for seg_start in candidate_starts:
            if seg_start >= info.start_idx and (seg_start + self.window_size) <= info.end_idx:
                seg, _ = self._extract_eeg_window(trial, seg_start, channel_last, trial_idx)
                slots.append(seg)
                if template is None:
                    template = seg
            else:
                if template is None:
                    slots.append(self._zero_eeg_window_tensor(channel_last, trial_idx))
                else:
                    slots.append(torch.zeros_like(template))
        return torch.stack(slots, dim=0)

    # ----- common slice helper -----

    def _extract_eeg_window(
        self,
        trial: _TrialCache,
        start: int,
        channel_last: bool,
        trial_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(eeg, eeg_padded)`` tensors for the given window.

        Normalization is controlled by ``self.normalize``:

        - ``window``: z-score using mean/std of the current window only.
        - ``trial``: z-score using precomputed trial-level mean/std.
        - ``subject``: z-score using per-subject mean/std from ``subject_stats/``.
        - ``none``: raw microvolt scale (float32).
        """
        end = start + self.window_size
        series = trial.eeg
        eeg_win = np.array(series[start:end], dtype=np.float32)

        if self.normalize == "window":
            pad_m = eeg_win.mean(axis=0)
            win_std = eeg_win.std(axis=0)
            pad_s = np.where(win_std < 1e-6, 1.0, win_std)
            eeg = (eeg_win - pad_m) / pad_s
        elif self.normalize == "none":
            eeg = eeg_win
            pad_m = np.zeros(eeg.shape[1], dtype=np.float32)
            pad_s = np.ones(eeg.shape[1], dtype=np.float32)
        else:
            mean_vec, std_vec = self._norm_mean_std_for_trial(trial, trial_idx)
            pad_m = np.asarray(mean_vec, dtype=np.float32)
            pad_s = np.where(np.asarray(std_vec, dtype=np.float32) < 1e-6,
                             1.0, np.asarray(std_vec, dtype=np.float32)).astype(np.float32)
            eeg = (eeg_win - pad_m) / pad_s

        if self.padding > 0:
            pad_start = max(0, start - self.padding)
            available = start - pad_start
            zero_needed = self.padding - available
            raw_padded = np.array(
                series[pad_start:end], dtype=np.float32,
            )
            if self.normalize != "none":
                raw_padded = (raw_padded - pad_m) / pad_s
            if zero_needed > 0:
                raw_padded = np.concatenate(
                    [np.zeros((zero_needed, eeg.shape[1]), dtype=np.float32),
                     raw_padded], axis=0,
                )
        else:
            raw_padded = eeg.copy()

        if channel_last:
            eeg_t = torch.from_numpy(np.ascontiguousarray(eeg)).clone()
            pad_t = torch.from_numpy(np.ascontiguousarray(raw_padded)).clone()
        else:
            eeg_t = torch.from_numpy(np.ascontiguousarray(eeg.T)).clone()
            pad_t = torch.from_numpy(np.ascontiguousarray(raw_padded.T)).clone()
        return eeg_t, pad_t

    def __len__(self) -> int:
        return len(self.index)


# ===================================================================
# SeedDataset  (multi-channel, 3-class emotion classification)
# ===================================================================

class SeedDataset(_BaseSeedDataset):
    """Dataset for SEED emotion recognition.

    Each sample yields an EEG window plus a 3-class label
    (0 = negative, 1 = neutral, 2 = positive).
    """

    def __init__(
        self,
        dataset_path: str,
        interval_length: int,
        subjects: Optional[Sequence[str]] = None,
        sessions: Optional[Sequence[int]] = None,
        trial_ids: Optional[Sequence[str]] = None,
        cache_size: int = 10,
        channel_last: bool = False,
        stride: Optional[int] = None,
        padding: int = 0,
        normalize: str = "window",
        subject_stats_dir: Optional[str] = None,
        freq_baseline_n_segments: int = 0,
    ) -> None:
        self.channel_last = channel_last
        super().__init__(
            dataset_path=dataset_path,
            interval_length=interval_length,
            subjects=subjects,
            sessions=sessions,
            trial_ids=trial_ids,
            cache_size=cache_size,
            stride=stride,
            padding=padding,
            normalize=normalize,
            subject_stats_dir=subject_stats_dir,
            freq_baseline_n_segments=freq_baseline_n_segments,
        )

    def _build_index(self, trial_idx: int, info: _TrialInfo) -> None:
        for start in range(
            info.start_idx,
            info.end_idx - self.window_size + 1,
            self.stride_size,
        ):
            self.index.append((trial_idx, start))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trial_idx, start = self.index[idx]
        trial = self._get_trial(trial_idx)
        info = self.trial_infos[trial_idx]

        eeg, eeg_padded = self._extract_eeg_window(
            trial, start, self.channel_last, trial_idx,
        )
        label = info.meta.get("label", 0)

        out: Dict[str, torch.Tensor] = {
            "eeg": eeg,
            "eeg_padded": eeg_padded,
            "trial": torch.tensor(trial_idx, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }
        if self.freq_baseline_n_segments > 0:
            out["eeg_history"] = self._eeg_history_stack(
                trial, start, trial_idx, self.channel_last,
            )
        return out


# ===================================================================
# Factory function
# ===================================================================

def create_seed_dataset(
    dataset_path: str,
    interval_length: int,
    subjects: Optional[Sequence[str]] = None,
    sessions: Optional[Sequence[int]] = None,
    trial_ids: Optional[Sequence[str]] = None,
    cache_size: int = 10,
    channel_last: bool = False,
    stride: Optional[int] = None,
    padding: int = 0,
    normalize: str = "window",
    subject_stats_dir: Optional[str] = None,
    freq_baseline_n_segments: int = 0,
) -> SeedDataset:
    return SeedDataset(
        dataset_path=dataset_path,
        interval_length=interval_length,
        subjects=subjects,
        sessions=sessions,
        trial_ids=trial_ids,
        cache_size=cache_size,
        channel_last=channel_last,
        stride=stride,
        padding=padding,
        normalize=normalize,
        subject_stats_dir=subject_stats_dir,
        freq_baseline_n_segments=freq_baseline_n_segments,
    )
