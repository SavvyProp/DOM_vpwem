"""Privileged tracking targets for the separate shuffle-touch VPWEM variant."""

from pathlib import Path

import numpy as np
import torch

from .data import DatasetFormatError, MikasaNpzDataset


def load_tracking_labels(path: Path, full_length: int, length: int):
    with np.load(path, allow_pickle=False) as archive:
        missing = {"tracking_xy", "tracking_hidden"}.difference(archive.files)
        if missing:
            raise DatasetFormatError(
                f"{path}: missing tracking labels {sorted(missing)}. Collect fresh "
                "1-2-swap demonstrations with dom_vpwem.collect_demos."
            )
        xy = np.asarray(archive["tracking_xy"])
        hidden = np.asarray(archive["tracking_hidden"])
        if xy.shape != (full_length, 2) or not np.issubdtype(xy.dtype, np.number):
            raise DatasetFormatError(f"{path}: tracking_xy must have shape [T,2]")
        if not np.isfinite(xy).all():
            raise DatasetFormatError(f"{path}: nonfinite tracking_xy")
        if hidden.shape != (full_length,) or not np.isin(hidden, [0, 1]).all():
            raise DatasetFormatError(f"{path}: tracking_hidden must be a boolean [T] array")
        return xy[:length].astype(np.float32), hidden[:length].astype(bool)


class TrackingNpzDataset(MikasaNpzDataset):
    """Add causal, timestamp-aligned labels; never put privileged state in obs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.memory_subsample_ratio != 1:
            raise ValueError("Tracking supervision requires memory_subsample_ratio=1")
        self.tracking_labels = tuple(
            load_tracking_labels(record.path, record.full_length, record.length)
            for record in self.episodes
        )

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        episode_index = int(sample["episode_index"])
        timestep = int(sample["timestep"])
        xy, hidden = self.tracking_labels[episode_index]
        obs_indices = np.maximum(np.arange(timestep - self.obs_steps + 1, timestep + 1), 0)
        history_indices = sample["memory_timestep"].numpy()
        indices = np.concatenate([np.maximum(history_indices, 0), obs_indices])
        mask = torch.cat([sample["memory_mask"], sample["obs_mask"]])
        sample["tracking_xy"] = self._tensor(xy[indices])
        sample["tracking_hidden"] = self._tensor(hidden[indices])
        sample["tracking_mask"] = mask
        return sample
