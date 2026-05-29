from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from .seed import _BaseSeedDataset, _TrialInfo


# ===================================================================
# SeedIVDataset  (multi-channel, 4-class emotion classification)
# ===================================================================


class SeedIVDataset(_BaseSeedDataset):
    """Dataset for SEED-IV emotion recognition.

    Each sample yields an EEG window plus a 4-class label
    (0 = neutral, 1 = sad, 2 = fear, 3 = happy).

    The preprocessed directory layout and file format are identical to
    other SEED-family datasets (SEED, SEED-V, SEED-VII), so the scanning,
    caching, and windowing logic from :class:`_BaseSeedDataset` is reused.
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


def create_seediv_dataset(
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
) -> SeedIVDataset:
    return SeedIVDataset(
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

