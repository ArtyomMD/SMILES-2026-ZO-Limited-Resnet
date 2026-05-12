"""
train_data.py — adaptive class-aware train loader.

The loader does not read saved confusion matrices. Instead, zo_optimizer.py sends
in-memory feedback through set_optimizer_feedback(). This allows the final
solution to keep all initialization/confusion computation inside head_init.py.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets

from augmentation import get_transforms


NUM_CLASSES = 100

# Conservative defaults matching the archived best run: no mosaics, mild
# confusion-aware sampling. These can be adjusted independently.
MOSAIC_ENABLED = os.getenv("MOSAIC_ENABLED", "0") != "0"
MOSAIC_GRID = int(os.getenv("MOSAIC_GRID", "2"))
MOSAIC_START_PROB = float(os.getenv("MOSAIC_START_PROB", "0.10"))
MOSAIC_DECAY_FRACTION = float(os.getenv("MOSAIC_DECAY_FRACTION", "0.35"))
MOSAIC_FOCUS_ONLY_PROB = float(os.getenv("MOSAIC_FOCUS_ONLY_PROB", "0.70"))

FOCUS_CLASS_BOOST = float(os.getenv("FOCUS_CLASS_BOOST", "1.0"))
HARDNESS_ALPHA = float(os.getenv("HARDNESS_ALPHA", "0.5"))
CONFUSER_BOOST = float(os.getenv("CONFUSER_BOOST", "0.5"))
MAX_CLASS_WEIGHT = float(os.getenv("MAX_CLASS_WEIGHT", "2.0"))
VIRTUAL_DATASET_LEN = int(os.getenv("VIRTUAL_DATASET_LEN", "50000"))

_LAST_BATCH_LABELS: torch.Tensor | None = None
_OPTIMIZER_CONFUSION: np.ndarray | None = None
_OPTIMIZER_FOCUS_CLASSES: list[int] = []
_OPTIMIZER_STEP: int = 0


def get_last_batch_labels() -> torch.Tensor | None:
    return _LAST_BATCH_LABELS


def set_optimizer_feedback(
    confusion_matrix: np.ndarray | None = None,
    focus_classes: Iterable[int] | None = None,
    step_idx: int | None = None,
) -> None:
    global _OPTIMIZER_CONFUSION, _OPTIMIZER_FOCUS_CLASSES, _OPTIMIZER_STEP

    if confusion_matrix is not None:
        cm = np.asarray(confusion_matrix, dtype=np.float64)
        if cm.shape == (NUM_CLASSES, NUM_CLASSES):
            _OPTIMIZER_CONFUSION = cm

    if focus_classes is not None:
        clean: list[int] = []
        for c in focus_classes:
            c = int(c)
            if 0 <= c < NUM_CLASSES:
                clean.append(c)
        _OPTIMIZER_FOCUS_CLASSES = sorted(set(clean))

    if step_idx is not None:
        _OPTIMIZER_STEP = int(step_idx)


def _record_last_batch_labels(labels: torch.Tensor) -> None:
    global _LAST_BATCH_LABELS
    _LAST_BATCH_LABELS = labels.detach().long().cpu()


def _read_cli_int(flag: str, default: int) -> int:
    argv = sys.argv
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
        if item.startswith(flag + "="):
            try:
                return int(item.split("=", 1)[1])
            except ValueError:
                return default
    return default


def _hardness_from_confusion(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    support = cm.sum(axis=1)
    correct = np.diag(cm)

    acc = np.full(NUM_CLASSES, 0.5, dtype=np.float64)
    valid = support > 0
    acc[valid] = correct[valid] / np.maximum(support[valid], 1e-12)

    hardness = 1.0 - acc
    if support.max() > 0:
        evidence = np.log1p(support) / (np.log1p(support).max() + 1e-12)
        hardness = 0.80 * hardness + 0.20 * evidence
    return acc, hardness


class AdaptiveCIFAR100(Dataset):
    """Dataset shell with a custom collate_fn that samples adaptive batches."""

    def __init__(self, data_dir: str, batch_size: int, generator_train: torch.Generator) -> None:
        self.base = datasets.CIFAR100(
            root=data_dir,
            train=True,
            download=True,
            transform=None,
        )
        self.targets = np.asarray(self.base.targets, dtype=np.int64)
        self.indices_by_class = [np.where(self.targets == c)[0] for c in range(NUM_CLASSES)]
        self.transform = get_transforms(train=True)
        self.batch_size = int(batch_size)

        seed = 42
        try:
            seed = int(generator_train.initial_seed())
        except Exception:
            pass
        self.rng = np.random.default_rng(seed)

        self.total_steps = max(1, _read_cli_int("--n_batches", default=64))
        self.local_batch_idx = 0

    def __len__(self) -> int:
        return VIRTUAL_DATASET_LEN

    def __getitem__(self, idx: int) -> int:
        return int(idx)

    def _class_probabilities(self) -> np.ndarray:
        weights = np.ones(NUM_CLASSES, dtype=np.float64)

        cm = _OPTIMIZER_CONFUSION
        if cm is not None and cm.shape == (NUM_CLASSES, NUM_CLASSES) and cm.sum() > 0:
            _, hardness = _hardness_from_confusion(cm)
            weights *= 1.0 + HARDNESS_ALPHA * hardness

            hard = np.argsort(hardness)[::-1][:24]
            for c in hard:
                row = cm[int(c)].copy()
                row[int(c)] = 0.0
                if row.sum() <= 0:
                    continue
                confusers = np.argsort(row)[::-1][:2]
                for j in confusers:
                    if row[int(j)] > 0:
                        weights[int(j)] += CONFUSER_BOOST

        if _OPTIMIZER_FOCUS_CLASSES:
            weights[np.asarray(_OPTIMIZER_FOCUS_CLASSES, dtype=np.int64)] += FOCUS_CLASS_BOOST

        weights = np.clip(weights, 1e-3, MAX_CLASS_WEIGHT)
        return weights / weights.sum()

    def _sample_class(self, prefer_focus: bool = False) -> int:
        if prefer_focus and _OPTIMIZER_FOCUS_CLASSES and self.rng.random() < MOSAIC_FOCUS_ONLY_PROB:
            return int(self.rng.choice(np.asarray(_OPTIMIZER_FOCUS_CLASSES, dtype=np.int64)))
        return int(self.rng.choice(np.arange(NUM_CLASSES), p=self._class_probabilities()))

    def _sample_index_from_class(self, cls: int) -> int:
        indices = self.indices_by_class[int(cls)]
        return int(self.rng.choice(indices))

    def _make_single(self, cls: int) -> torch.Tensor:
        idx = self._sample_index_from_class(cls)
        img, _ = self.base[idx]
        return self.transform(img)

    def _make_mosaic(self, cls: int) -> torch.Tensor:
        grid = int(MOSAIC_GRID)
        tiles = [self._make_single(cls) for _ in range(grid * grid)]
        rows = []
        for r in range(grid):
            rows.append(torch.cat(tiles[r * grid : (r + 1) * grid], dim=2))
        return torch.cat(rows, dim=1)

    def _mosaic_probability(self) -> float:
        if not MOSAIC_ENABLED:
            return 0.0
        decay_steps = max(1, int(self.total_steps * MOSAIC_DECAY_FRACTION))
        progress = min(1.0, self.local_batch_idx / float(decay_steps))
        return float(MOSAIC_START_PROB * (1.0 - progress) ** 2)

    def collate_fn(self, dummy_batch: list[int]):
        del dummy_batch
        use_mosaic = self.rng.random() < self._mosaic_probability()

        images: list[torch.Tensor] = []
        labels: list[int] = []
        for _ in range(self.batch_size):
            cls = self._sample_class(prefer_focus=use_mosaic)
            image = self._make_mosaic(cls) if use_mosaic else self._make_single(cls)
            images.append(image)
            labels.append(cls)

        labels_t = torch.tensor(labels, dtype=torch.long)
        _record_last_batch_labels(labels_t)
        self.local_batch_idx += 1
        return torch.stack(images, dim=0), labels_t


def get_train_dataset_loader(data_dir, batch_size, generator_train):
    train_dataset = AdaptiveCIFAR100(
        data_dir=data_dir,
        batch_size=batch_size,
        generator_train=generator_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        generator=generator_train,
        collate_fn=train_dataset.collate_fn,
    )

    return train_dataset, train_loader
