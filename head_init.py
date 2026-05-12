"""
head_init.py — preset-aware final-layer initialization.

This file computes all head initializations and the initial confusion matrix in
memory. No new_fc.pth / confusion_matrix.npy files are written or required.

The active preset is read from zo_optimizer.SOLUTION_PRESET.
"""

from __future__ import annotations

from collections import defaultdict
import os
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.models as models

from augmentation import get_transforms


NUM_CLASSES = 100
FEATURE_DIM = 512

# Preset-specific initialization knobs. They are intentionally independent from
# optimizer knobs so experiments can be changed one piece at a time.
# Use 500 for full CIFAR100 train split; use 80 for an 8000-sample balanced set.
INIT_SAMPLES_PER_CLASS = {
    1: 0,     # Xavier does not need data.
    2: 80,    # Hungarian matching on a balanced 8000-sample subset.
    3: 80,    # Ridge on a balanced 8000-sample subset. Increase to 500 if desired.
    4: 500,   # Final/default: reproduce current strong ridge initialization.
}

INIT_BATCH_SIZE = int(os.getenv("INIT_BATCH_SIZE", "32"))
INIT_SEED = int(os.getenv("INIT_SEED", "42"))
RIDGE_L2 = float(os.getenv("RIDGE_L2", "1e-2"))

_INITIAL_CONFUSION_MATRIX: np.ndarray | None = None
_INITIAL_MATCHING: np.ndarray | None = None
_LAST_INIT_PRESET: int | None = None


def _get_solution_preset() -> int:
    try:
        import zo_optimizer

        return int(getattr(zo_optimizer, "SOLUTION_PRESET", 4))
    except Exception:
        return int(os.getenv("SOLUTION_PRESET", "4"))


def _read_cli_value(flag: str, default: str) -> str:
    argv = sys.argv
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return default


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_initial_confusion_matrix() -> np.ndarray | None:
    """Called by zo_optimizer.py after init_last_layer() has run."""
    if _INITIAL_CONFUSION_MATRIX is None:
        return None
    return _INITIAL_CONFUSION_MATRIX.copy()


def get_initial_matching() -> np.ndarray | None:
    if _INITIAL_MATCHING is None:
        return None
    return _INITIAL_MATCHING.copy()


def _balanced_subset(dataset: datasets.CIFAR100, samples_per_class: int, seed: int) -> Subset:
    if samples_per_class <= 0 or samples_per_class >= 500:
        return Subset(dataset, list(range(len(dataset))))

    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(dataset.targets):
        by_class[int(label)].append(int(idx))

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in range(NUM_CLASSES):
        candidates = np.asarray(by_class[cls], dtype=np.int64)
        if len(candidates) < samples_per_class:
            raise ValueError(f"Class {cls} has only {len(candidates)} samples")
        chosen = rng.choice(candidates, size=samples_per_class, replace=False)
        selected.extend(int(x) for x in chosen)

    rng.shuffle(selected)
    return Subset(dataset, selected)


def _make_loader(samples_per_class: int) -> DataLoader:
    data_dir = _read_cli_value("--data_dir", "./data")
    base = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=False),
    )
    subset = _balanced_subset(base, samples_per_class=samples_per_class, seed=INIT_SEED)
    return DataLoader(
        subset,
        batch_size=INIT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def _load_imagenet_resnet(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.to(device)
    model.eval()
    return model


def _linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian algorithm replacement for scipy.optimize.linear_sum_assignment.

    Supports rectangular matrices with rows <= cols; transposes otherwise.
    Returns row_ind, col_ind minimizing cost_matrix[row_ind, col_ind].
    """
    cost = np.asarray(cost_matrix, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2D")

    n0, m0 = cost.shape
    transposed = False
    if n0 > m0:
        cost = cost.T
        transposed = True

    n, m = cost.shape
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    if not transposed:
        row_ind = np.arange(n, dtype=np.int64)
        col_ind = assignment
    else:
        row_ind = assignment
        col_ind = np.arange(n, dtype=np.int64)
        order = np.argsort(row_ind)
        row_ind = row_ind[order]
        col_ind = col_ind[order]

    return row_ind, col_ind


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for y, p in zip(y_true, y_pred):
        yi = int(y)
        pi = int(p)
        if 0 <= yi < NUM_CLASSES and 0 <= pi < NUM_CLASSES:
            cm[yi, pi] += 1.0
    return cm


def _collect_imagenet_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="  Initial inference"):
            images = images.to(device, non_blocking=True)
            logits = model(images).detach().cpu()
            logits_all.append(logits.float())
            labels_all.append(labels.cpu().long())
    return torch.cat(logits_all, dim=0), torch.cat(labels_all, dim=0)


def _collect_features(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    backbone = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    x_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="  Initial inference"):
            images = images.to(device, non_blocking=True)
            z = backbone(images).flatten(1).detach().cpu().numpy().astype(np.float64)
            x_all.append(z)
            y_all.append(labels.numpy().astype(np.int64))

    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)


def _fit_ridge_fc(X: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    Y = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    Y[np.arange(n), y] = 1.0

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xs = (X - mean) / std
    Xb = np.concatenate([np.ones((n, 1), dtype=np.float64), Xs], axis=1)

    reg = np.eye(d + 1, dtype=np.float64)
    reg[0, 0] = 0.0
    Wb = np.linalg.solve(Xb.T @ Xb + l2 * reg, Xb.T @ Y)

    bias_std = Wb[0]
    weight_std = Wb[1:]
    weight = weight_std / std[:, None]
    bias = bias_std - (mean / std) @ weight_std
    return weight.T.astype(np.float32), bias.astype(np.float32)


def _xavier_init(layer: nn.Linear) -> None:
    global _INITIAL_CONFUSION_MATRIX
    with torch.no_grad():
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    _INITIAL_CONFUSION_MATRIX = None


def _hungarian_init(layer: nn.Linear, samples_per_class: int) -> None:
    global _INITIAL_CONFUSION_MATRIX, _INITIAL_MATCHING
    device = _device()
    loader = _make_loader(samples_per_class)
    model = _load_imagenet_resnet(device)

    logits_all, labels_all = _collect_imagenet_logits(model, loader, device)
    scores = torch.zeros(NUM_CLASSES, 1000, dtype=torch.float32)
    for cls in range(NUM_CLASSES):
        mask = labels_all == cls
        if bool(mask.any()):
            scores[cls] = logits_all[mask].mean(dim=0)

    row_ind, col_ind = _linear_sum_assignment(-scores.numpy())
    matching = np.empty(NUM_CLASSES, dtype=np.int64)
    matching[row_ind] = col_ind
    _INITIAL_MATCHING = matching.copy()

    old_fc = model.fc.cpu()
    with torch.no_grad():
        idx = torch.as_tensor(matching, dtype=torch.long)
        layer.weight.copy_(old_fc.weight[idx])
        layer.bias.copy_(old_fc.bias[idx])

    logits_100 = logits_all[:, torch.as_tensor(matching, dtype=torch.long)]
    preds = logits_100.argmax(dim=1).numpy()
    _INITIAL_CONFUSION_MATRIX = _confusion_matrix(labels_all.numpy(), preds)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ridge_init(layer: nn.Linear, samples_per_class: int) -> None:
    global _INITIAL_CONFUSION_MATRIX
    device = _device()
    loader = _make_loader(samples_per_class)
    model = _load_imagenet_resnet(device)

    X, y = _collect_features(model, loader, device)
    weight, bias = _fit_ridge_fc(X, y, l2=RIDGE_L2)

    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(weight))
        layer.bias.copy_(torch.from_numpy(bias))

    logits = X @ weight.T.astype(np.float64) + bias.astype(np.float64)[None, :]
    preds = logits.argmax(axis=1)
    _INITIAL_CONFUSION_MATRIX = _confusion_matrix(y, preds)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def init_last_layer(layer: nn.Linear) -> None:

    """Initialize the new CIFAR100 head in-place according to SOLUTION_PRESET."""
    global _LAST_INIT_PRESET
    preset = _get_solution_preset()
    _LAST_INIT_PRESET = preset

    samples_per_class = int(INIT_SAMPLES_PER_CLASS.get(preset, 80))

    if preset == 1:
        _xavier_init(layer)
    elif preset == 2:
        _hungarian_init(layer, samples_per_class=samples_per_class)
    elif preset in (3, 4):
        _ridge_init(layer, samples_per_class=samples_per_class)
    else:
        raise ValueError(f"Unknown SOLUTION_PRESET={preset}")
